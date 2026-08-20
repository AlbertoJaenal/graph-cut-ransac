"""Cambridge Landmarks dataset utilities for model scaling and evaluation."""
import logging

import cv2
import numpy as np

from hloc.utils.read_write_model import (
    qvec2rotmat,
    read_cameras_binary,
    read_cameras_text,
    read_images_binary,
    read_images_text,
    read_model,
    write_model,
)
import imagesize

logger = logging.getLogger(__name__)


def scale_sfm_images(full_model, scaled_model, image_dir):
    """Scale COLMAP camera intrinsics to match original image resolution."""
    logger.info("Scaling the COLMAP model to the original image size.")
    scaled_model.mkdir(exist_ok=True)
    cameras, images, points3D = read_model(full_model)

    scaled_cameras = {}
    for id_, image in images.items():
        name = image.name
        img = cv2.imread(str(image_dir / name))
        assert img is not None, image_dir / name
        h, w = img.shape[:2]

        cam_id = image.camera_id
        if cam_id in scaled_cameras:
            assert scaled_cameras[cam_id].width == w
            assert scaled_cameras[cam_id].height == h
            continue

        camera = cameras[cam_id]
        assert camera.model == "SIMPLE_RADIAL"
        sx = w / camera.width
        sy = h / camera.height
        assert sx == sy, (sx, sy)
        scaled_cameras[cam_id] = camera._replace(
            width=w, height=h, params=camera.params * np.array([sx, sx, sy, 1.0])
        )

    write_model(scaled_cameras, images, points3D, scaled_model)


def create_query_list_with_intrinsics(
    model, out, list_file=None, ext=".bin", image_dir=None
):
    """Create query list file with camera intrinsics from COLMAP model."""
    if ext == ".bin":
        images = read_images_binary(model / "images.bin")
        cameras = read_cameras_binary(model / "cameras.bin")
    else:
        images = read_images_text(model / "images.txt")
        cameras = read_cameras_text(model / "cameras.txt")

    name2id = {image.name: i for i, image in images.items()}
    if list_file is None:
        names = list(name2id)
    else:
        with open(list_file, "r") as f:
            names = f.read().rstrip().split("\n")
    data = []
    for name in names:
        image = images[name2id[name]]
        camera = cameras[image.camera_id]
        w, h, params = camera.width, camera.height, camera.params

        if image_dir is not None:
            # Check the original image size and rescale the camera intrinsics
            img = cv2.imread(str(image_dir / name))
            assert img is not None, image_dir / name
            h_orig, w_orig = img.shape[:2]
            assert camera.model == "SIMPLE_RADIAL"
            sx = w_orig / w
            sy = h_orig / h
            assert sx == sy, (sx, sy)
            w, h = w_orig, h_orig
            params = params * np.array([sx, sx, sy, 1.0])

        p = [name, camera.model, w, h] + params.tolist()
        data.append(" ".join(map(str, p)))
    with open(out, "w") as f:
        f.write("\n".join(data))


def evaluate_cambridge(model, results, image_dir, list_file=None, ext=".bin", only_localized=False, out_path=None):
    predictions = {}
    with open(results, "r") as f:
        for data in f.read().rstrip().split("\n"):
            data = data.split()
            name = data[0]
            q, t = np.split(np.array(data[1:-2], float), [4])
            f, tim = np.array(data[-2:], float)
            predictions[name] = (qvec2rotmat(q), t, f, tim)
    if ext == ".bin":
        images = read_images_binary(model / "images.bin")
        cameras = read_cameras_binary(model / "cameras.bin")
    else:
        images = read_images_text(model / "images.txt")
        cameras = read_cameras_text(model / "cameras.txt")
    name2id = {image.name: i for i, image in images.items()}

    if list_file is None:
        test_names = list(name2id)
    else:
        with open(list_file, "r") as f:
            test_names = f.read().rstrip().split("\n")

    errors_t = []
    errors_R = []
    errors_f = []
    tims = []
    N = 0
    for name in test_names:
        if name not in predictions:
            if only_localized:
                continue
            e_t = np.inf
            e_R = 180.0
        else:
            image = images[name2id[name]]
            R_gt, t_gt = image.qvec2rotmat(), image.tvec
            R, t, f, tim = predictions[name]
            e_t = np.linalg.norm(-R_gt.T @ t_gt + R.T @ t, axis=0)
            cos = np.clip((np.trace(np.dot(R_gt.T, R)) - 1) / 2, -1.0, 1.0)
            e_R = np.rad2deg(np.abs(np.arccos(cos)))
            if not np.allclose(t, np.zeros([3])) and not np.any(np.isinf(t)) and not np.any(np.isnan(t)): 
                N+=1
            else:
                continue

            w, h, params = cameras[image.camera_id].width, cameras[image.camera_id].height, cameras[image.camera_id].params
            assert (image_dir / name).exists()
            w_orig, h_orig = imagesize.get(str(image_dir / name))
            assert cameras[image.camera_id].model == "SIMPLE_RADIAL"
            sx, sy = w_orig / w, h_orig / h
            assert sx == sy, (sx, sy)
            w, h = w_orig, h_orig

            f_gt = params[0] * sx
            e_f = np.abs(f_gt - f) / f_gt
        errors_t.append(e_t)
        errors_R.append(e_R)
        errors_f.append(e_f)
        tims.append(tim)

    errors_t = np.array(errors_t)
    errors_R = np.array(errors_R)
    errors_f = np.array(errors_f)
    tims = np.array(tims)

    med_t = np.median(errors_t)
    med_R = np.median(errors_R)
    med_f = np.median(errors_f)
    med_tim = np.median(tims)*1e3
    out = f"cm,deg,f,tim,loc,total"
    result = ""
    result += f"\n{med_t*100:.3f},{med_R:.3f},{med_f:.3f},{med_tim:.3f},{N},{len(test_names)}"

    # out += "\nPercentage of test images localized within:"
    threshs_t = [0.01, 0.02, 0.03, 0.05, 0.25, 0.5, 5.0]
    threshs_R = [1.0, 2.0, 3.0, 5.0, 2.0, 5.0, 10.0]
    for th_t, th_R in zip(threshs_t, threshs_R):
        ratio = np.mean((errors_t < th_t) & (errors_R < th_R))
        out += f",{th_t*100:.0f}cm-{th_R:.0f}deg"
        result += f",{ratio*100:.2f}"
    
    print(out, result)
    if out_path is not None:
        with open(out_path, "w+") as f_:
            f_.write(out + result)