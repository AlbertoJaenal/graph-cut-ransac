#pragma once

#include <Eigen/Dense>
#include <vector>

void affine2sift_solver(Eigen::MatrixXd const& A, double q1, std::vector<double>* r1, std::vector<double>* r2)
{
    // Compute coefficients
    double const* A_data = A.data();
    double const A11 = A_data[0];
    double const A21 = A_data[1];
    double const A12 = A_data[2];
    double const A22 = A_data[3];
    Eigen::VectorXd coeffs(14);
    double _t2_ = A12*2.0;
    double _t3_ = A22*2.0;
    double _t4_ = q1*2.0;
    double _t5_ = -A11;
    double _t6_ = -A21;
    double _t7_ = -q1;
    double _t8_ = -_t4_;
    coeffs[0] = q1+_t5_;
    coeffs[1] = _t5_+_t7_;
    coeffs[2] = _t2_;
    coeffs[3] = _t2_;
    coeffs[4] = A11+q1;
    coeffs[5] = A11+_t7_;
    coeffs[6] = _t6_;
    coeffs[7] = _t8_;
    coeffs[8] = _t6_;
    coeffs[9] = _t3_;
    coeffs[10] = _t3_;
    coeffs[11] = A21;
    coeffs[12] = _t8_;
    coeffs[13] = A21;


    // Setup elimination template
    static const int coeffs0_ind[] = { 0,6,0,6,7,0,2,6,9,1,7,8,1,8,1,3,7,8,10,0,2,6,7,9,2,4,9,11 };
    static const int coeffs1_ind[] = { 5,13,3,5,10,13,1,3,8,10,3,5,10,12,13,2,4,9,11,12,5,12,13,4,11,12,4,11 };
        

    static const int C0_ind[] = {3,7,10,14,15,17,19,21,23,27,30,31,34,38,41,43,44,45,47,48,50,52,53,54,57,59,61,63};

    static const int C1_ind[] = {0,4,8,10,12,14,16,18,20,22,25,27,29,30,31,32,34,36,38,39,41,44,45,48,52,53,57,61};

    Eigen::MatrixXd C0 = Eigen::MatrixXd::Zero(8,8);
    Eigen::MatrixXd C1 = Eigen::MatrixXd::Zero(8,8);
    for (int i = 0; i < 28; i++) {
        C0(C0_ind[i]) = coeffs(coeffs0_ind[i]);
    }

    for (int i = 0; i < 28; i++) {
        C1(C1_ind[i]) = coeffs(coeffs1_ind[i]);
    }

    Eigen::MatrixXd C12 = C0.fullPivLu().solve(C1);



    // Setup action matrix
    Eigen::Matrix<double,12, 8> RR;
    RR << -C12.bottomRows(4), Eigen::Matrix<double,8,8>::Identity(8, 8);

    static const int AM_ind[] = { 5,6,0,1,2,7,8,3 };
    Eigen::Matrix<double, 8, 8> AM;
    for (int i = 0; i < 8; i++) {
        AM.row(i) = RR.row(AM_ind[i]);
    }

    Eigen::MatrixXcd sols(2, 8);
    sols.setZero();

    // Solve eigenvalue problem
    Eigen::EigenSolver<Eigen::Matrix<double, 8, 8> > es(AM);
    Eigen::ArrayXcd D = es.eigenvalues();    
    Eigen::ArrayXXcd V = es.eigenvectors();

    V = (V / V.row(0).array().replicate(8, 1)).eval();


    sols.row(0) = D.transpose().array();
    sols.row(1) = V.row(5).array();




    Eigen::VectorXcd r1c = sols.row(0);
    Eigen::VectorXcd r2c = sols.row(1);
    int nsols = r1c.size();
    for (int isol = 0; isol < nsols; ++isol) {
        if ( std::abs(r1c(isol).imag()) < 1e-6 && std::abs(r2c(isol).imag()) < 1e-6 )
        {
            r1->push_back(r1c(isol).real());
            r2->push_back(r2c(isol).real());
        }
    }
}


void affine2sift(const Eigen::Matrix2d &A, double &s1, double &c1, double &s2, double &c2, double &q )
{
    q = sqrt(A.determinant());
    std::vector<double> r1solns, r2solns;
    affine2sift_solver(A, q, &r1solns, &r2solns);
    
    if (r1solns.empty()) {
        // Handle error or use a default identity decomposition
        c1 = c2 = 1.0; s1 = s2 = 0.0; return;
    } else {
        double r1 = r1solns[0];
        double r2 = r2solns[0];
        c1 = (1-r1*r1)/(1+r1*r1);
        s1 = (2*r1)/(1+r1*r1);
        c2 = (1-r2*r2)/(1+r2*r2);
        s2 = (2*r2)/(1+r2*r2);
    }
    
    // check residuals
    //double res1 = c1*s2*A(0,0) + s1*s2*A(0,1) - c1*c2*A(1,0) - c2*s1*A(1,1);
    //double res2 = A(0,1)*A(1,0)-A(0,0)*A(1,1)+q*q;
    //double res3 = A(0,0)*c1 + A(0,1)*s1 - c2*q;
    //double res4 = A(1,0)*c1 + A(1,1)*s1 - s2*q;
}