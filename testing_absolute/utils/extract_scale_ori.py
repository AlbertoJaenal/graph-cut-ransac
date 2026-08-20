"""Extract scale and orientation information from features using S3Esti.

Provides functionality to estimate SIFT-like scale and orientation from feature
keypoints, and save Local Affine Frames (LAFs) to disk for later use in absolute
pose estimation.
"""
import sys
import os
from pathlib import Path
import numpy as np
import torch
from PIL import Image
import cv2
import h5py
from tqdm import tqdm

from hloc.utils.io import get_keypoints, list_h5_names
import kornia
from kornia.image import image_to_tensor

thirparty_path = Path(__file__).parent / '../thirdparty'
se3path = str(thirparty_path / 'S3Esti')
sys.path.append(se3path)
from abso_esti_model.abso_esti_net import EstiNet


def point2patch(coord, img, patch_size):
    """Extract patches around keypoints from an image.
    
    Args:
        coord: Keypoint coordinates (batch, n_keypoints, 4, 2) for 4 corners
        img: Input image (grayscale or RGB)
        patch_size: Size of extracted patches
    
    Returns:
        Tensor of patches (batch, n_keypoints, patch_size, patch_size)
    """
    # Convert to tensor and move to device
    if isinstance(img, Image.Image):
        img = torch.from_numpy(np.array(img))
    else:
        img = torch.as_tensor(img)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    img = img.to(device)
    coord = coord.to(device)

    # Ensure correct batch and point dimensions
    if coord.ndim == 3:
        coord = coord.unsqueeze(0)
    if img.ndim == 2:
        img = img.unsqueeze(0)

    B, N, _, _ = coord.shape
    output_batches = []

    for b in range(B):
        current_batch_patches = []
        for n in range(N):
            pts = coord[b, n]
            
            # Extract patch bounds from corner coordinates
            ymin, xmin = pts.min(dim=0).values.long()
            ymax, xmax = pts.max(dim=0).values.long()
            patch = img[b, xmin:xmax, ymin:ymax]

            # Handle size mismatch: pad or zero if needed
            if patch.shape[-2:] != (patch_size, patch_size):
                patch = torch.zeros((patch_size, patch_size), device=device)
            
            current_batch_patches.append(patch)
        
        output_batches.append(torch.stack(current_batch_patches))

    return torch.stack(output_batches)

def extract_affine_patches(kpts, scales, rotations, in_radians=True):
    """Build affine-transformed patch descriptors (Local Affine Frames).
    
    Creates 2x2 matrices representing scale and rotation for each keypoint,
    used for affine-consistency checks in absolute pose estimation.
    
    Args:
        kpts: Keypoint coordinates (N, 2) [x, y]
        scales: SIFT scales (N,) - scale parameter sigma
        rotations: SIFT orientations (N,) - in degrees or radians
        in_radians: If True, rotations are already in radians
    
    Returns:
        (N, 2, 2) array of affine transformation matrices
    """
    # Convert to radians if needed
    rad = rotations if in_radians else np.radians(rotations)
    cos = np.cos(rad)
    sin = np.sin(rad)
    
    # Build isotropic rotation+scale matrices
    M = np.zeros((len(kpts), 2, 2))
    M[:, 0, 0] = scales * cos      # s*cos
    M[:, 0, 1] = scales * (-sin)   # -s*sin
    M[:, 1, 0] = scales * sin      # s*sin
    M[:, 1, 1] = scales * cos      # s*cos
    
    return M

class S3Esti:
    """S3Esti neural network for scale and orientation estimation.
    
    Estimates SIFT-like scale and orientation from local image patches using
    two separate EstiNet models (one for scale, one for orientation).
    """
    
    def __init__(self, device):
        """Initialize S3Esti models.
        
        Args:
            device: PyTorch device ("cuda" or "cpu")
        """
        patch_size = 32
        scale_num = 300
        angle_num = 360
        esti_checkpoint_path = os.path.join(
            se3path, 'abso_esti_model/S3Esti_ep30.pth')
        self.device = torch.device(device)
        self.esti_scale_ratio_list = [0.5, 1, 2]

        # Initialize scale and orientation estimation networks
        self.model_scale = EstiNet(
            need_bn=True, device=device, out_channels=scale_num,
            patch_size=patch_size, scale_ratio=self.esti_scale_ratio_list)
        self.model_angle = EstiNet(
            need_bn=True, device=device, out_channels=angle_num,
            patch_size=patch_size, scale_ratio=self.esti_scale_ratio_list)
        
        # Load pretrained checkpoint
        checkpoint = torch.load(esti_checkpoint_path, map_location=device)
        checkpoint['model_scale'].pop("base")
        checkpoint['model_angle'].pop("base")

        self.model_scale.load_state_dict(checkpoint['model_scale'], strict=True)
        self.model_scale.eval().to(device)
        self.model_angle.load_state_dict(checkpoint['model_angle'], strict=True)
        self.model_angle.eval().to(device)

        # Load shape affinity network
        self.affnet = kornia.feature.LAFAffNetShapeEstimator(True)
        self.affnet.eval().to(device)


    def generate_patch(self, img, coord_, patch_size):
        """Generate patches around keypoints for feature extraction.
        
        Args:
            img: Grayscale image
            coord_: Keypoint coordinates (N, 2)
            patch_size: Size of patches to extract
        
        Returns:
            Tensor of patches (N, patch_size, patch_size)
        """
        coord = torch.tensor(coord_).clone()
        
        # Clamp coordinates to image bounds
        PS = patch_size // 2
        coord[..., 0] = torch.clamp(coord[..., 0], min=PS, max=img.shape[1]-1-PS)
        coord[..., 1] = torch.clamp(coord[..., 1], min=PS, max=img.shape[0]-1-PS)

        # Get four corner coordinates around each keypoint
        tl, tr = torch.clone(coord), torch.clone(coord)
        bl, br = torch.clone(coord), torch.clone(coord)
        
        tl[:, 0] -= PS
        tl[:, 1] -= PS
        tr[:, 0] -= PS
        tr[:, 1] += PS
        bl[:, 0] += PS
        bl[:, 1] -= PS
        br[:, 0] += PS
        br[:, 1] += PS
        
        patch_coord = torch.cat(
            [tl.unsqueeze(1), tr.unsqueeze(1), bl.unsqueeze(1), br.unsqueeze(1)],
            dim=1)
        patch_tensor = point2patch(patch_coord, img, patch_size)

        # Filter out zero patches
        patch_ = [torch.empty(0, patch_tensor.shape[2], patch_tensor.shape[3])]
        for p in patch_tensor[0]:
            if not (p == 0).all():
                patch_.append(p.unsqueeze(0))

        patch_ = torch.cat(patch_, dim=0)
        if torch.cuda.is_available():
            patch_ = patch_.cuda()

        return patch_
    def add_scale_oris(self, data, pred):
        """Estimate scale and orientation for detected keypoints.
        
        Args:
            data: Dictionary with 'image' key (RGB tensor)
            pred: Dictionary with 'keypoints' key (tensor of shape [1, N, 2])
        
        Returns:
            Updated pred dictionary with 'scales' and 'oris' (in degrees)
        """
        kpts = pred['keypoints'][0].detach().cpu().numpy()
        assert data['image'].max() > 1, f"Image range: [{data['image'].min()}, {data['image'].max()}]"
        co = pred['keypoints'].numpy()
        
        # Convert to grayscale for patch extraction
        gray = cv2.cvtColor(data['image'].cpu().numpy(), cv2.COLOR_RGB2GRAY)
        patch_ = self.generate_patch(gray, co[0].round().astype(int), patch_size=32)
        
        img = image_to_tensor(data['image'].cpu().numpy(), False).float() / 255.
        img = kornia.color.bgr_to_rgb(img)
        
        # Expand for multi-scale processing (9 scales)
        patch_ = patch_.unsqueeze(1).expand(-1, 9, -1, -1)

        # Infer scale and orientation
        with torch.no_grad():
            scale_resp = self.model_scale(patch_)
            angle_resp = self.model_angle(patch_)

        # Get argmax predictions and convert to continuous values
        scale_ind_pred = torch.argmax(scale_resp, dim=1).cpu().numpy()
        angle_ind_pred = torch.argmax(angle_resp, dim=1).cpu().numpy()

        scales = self.model_scale.map_id_to_scale(scale_ind_pred).cpu().numpy()
        oris = self.model_angle.map_id_to_angle(angle_ind_pred).cpu().numpy()
        
        assert (oris.min() >= -np.pi - 0.01 and oris.max() <= np.pi + 0.01), \
            f"Orientation range error: [{oris.min()}, {oris.max()}]"

        pred['scales'] = scales
        pred['oris'] = oris / np.pi * 180  # Convert to degrees
        
        assert len(pred["scales"].squeeze()) == len(kpts) and \
               len(pred["oris"].squeeze()) == len(kpts), \
               f"Length mismatch: kpts={len(kpts)}, scales={len(pred['scales'].squeeze())}, oris={len(pred['oris'].squeeze())}"
        
        return pred

def save_laf(lafs_file: Path, qnam: str, lafs: np.ndarray, scales: np.ndarray, oris: np.ndarray):
    """Save Local Affine Frames and metadata to h5 file.
    
    Args:
        lafs_file: Output h5 file path
        qnam: Image name (group key)
        lafs: (N, 2, 2) affine transformation matrices
        scales: (N,) scale values
        oris: (N,) orientation values (in degrees)
    """
    with h5py.File(lafs_file, "r+" if os.path.exists(lafs_file) else "w") as f:
        g = f.create_group(qnam)
        g.create_dataset("lafs", data=lafs)
        g.create_dataset("scales", data=scales)
        g.create_dataset("oris", data=oris)

def get_scale_ori(path: Path, name: str) -> tuple:
    """Load scale and orientation from h5 file.
    
    Args:
        path: Path to features h5 file
        name: Image name
    
    Returns:
        (scales, oris) arrays
    """
    with h5py.File(str(path), "r", libver="latest") as hfile:
        s = hfile[name]["scales"].__array__()
        o = hfile[name]["oris"].__array__()
    return s, o


def extract_scale_ori(feat_h5_file, image_path, requires_sso_extract, image_list=None,
                      best_point_is="ori"):
    """Extract and save scale/orientation for all keypoints in feature file.
    
    Estimates scale and orientation using either S3Esti neural network (if
    requires_sso_extract=True) or retrieves from pre-computed values.
    
    Args:
        feat_h5_file: Path to input features h5 file (from hloc)
        image_path: Path to directory containing images
        requires_sso_extract: If True, use S3Esti to estimate; else retrieve
        image_list: Optional list of image names (uses all if None)
        best_point_is: Unused parameter (kept for API compatibility)
    """
    if image_list is None:
        image_list = list_h5_names(feat_h5_file)
    
    lafs_file = str(feat_h5_file).replace(".h5", "_lafs.h5")
    
    # Find which images already have LAFs computed
    existing_images = list_h5_names(lafs_file) if os.path.exists(lafs_file) else []
    to_do_images = list(set(image_list) - set(existing_images))

    # Initialize S3Esti for scale/orientation estimation
    s3esti = S3Esti("cuda" if torch.cuda.is_available() else "cpu")

    # Process each image
    for qnam in tqdm(to_do_images):
        kpts = get_keypoints(feat_h5_file, qnam)
        
        if not requires_sso_extract:
            # Use pre-computed scale/orientation
            s, o = get_scale_ori(feat_h5_file, qnam)
        else:
            # Estimate scale/orientation using S3Esti
            rgb = cv2.cvtColor(cv2.imread(str(Path(image_path) / qnam)), cv2.COLOR_BGR2RGB)
            data = {"image": torch.from_numpy(rgb).to(s3esti.device)}
            pred = {"keypoints": torch.from_numpy(kpts[None])}
            pred_extended = s3esti.add_scale_oris(data, pred)

            o = pred_extended['oris'].reshape([-1])
            s = pred_extended['scales'].reshape([-1])

        # Build Local Affine Frame matrices (isotropic: scale * rotation)
        lafs = extract_affine_patches(kpts, s, o, in_radians=False)
        
        # Save to disk
        save_laf(lafs_file, qnam, lafs, scales=s, oris=o)

    print(f" - Created {lafs_file}")