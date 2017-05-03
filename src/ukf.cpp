#include "ukf.h"
#include "tools.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF() {
  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;

  // initial state vector
  x_ = VectorXd(5);

  // initial covariance matrix
  P_ = MatrixXd(5, 5);
  P_ << 1, 0, 0, 0, 0,
		0, 1, 0, 0, 0,
		0, 0, 1, 0, 0,
		0, 0, 0, 1, 0,
		0, 0, 0, 0, 1;

  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 0.9;

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = 2.5;

  // Laser measurement noise standard deviation x in m
  std_laspx_ = 0.15;

  // Laser measurement noise standard deviation y in m
  std_laspy_ = 0.15;

  // Radar measurement noise standard deviation radius in m
  std_radr_ = 0.3;

  // Radar measurement noise standard deviation angle in rad
  std_radphi_ = 0.03;

  // Radar measurement noise standard deviation radius change in m/s
  std_radrd_ = 0.3;

  /**
  TODO:

  Complete the initialization. See ukf.h for other member properties.

  Hint: one or more values initialized above might be wildly off...
  */
  is_initialized_ = false;

  n_x_ = 5;

  n_w_ = 15;

  Xsig_pred_ = MatrixXd(n_x_, n_w_);

  time_us_ = 0;

  n_aug_ = 7;

  n_sigma_ = 2 * n_aug_ + 1;

  lambda_ = 3 - n_aug_;

  InitializeWeights();

  NIS_radar_ = 0;

  NIS_laser_ = 0;
}

UKF::~UKF() {}

/**
 * @param {MeasurementPackage} meas_package The latest measurement data of
 * either radar or laser.
 */
void UKF::ProcessMeasurement(const MeasurementPackage &meas_package) {
  if (!is_initialized_) {
	  time_us_ = meas_package.timestamp_;

    if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
      InitializeRadar(meas_package);
    }
    else if (meas_package.sensor_type_ == MeasurementPackage::LASER) {
      InitializeLaser(meas_package);
    }

    is_initialized_ = true;
    return;
  }

  double dt = (meas_package.timestamp_ - time_us_) / 1000000.0;
	time_us_ = meas_package.timestamp_;

	Prediction(dt);
	
  if (meas_package.sensor_type_ == MeasurementPackage::RADAR && use_radar_) {
		UpdateRadar(meas_package);
	}
	else if(meas_package.sensor_type_ == MeasurementPackage::LASER && use_laser_) {
		UpdateLidar(meas_package);
	}	
}

void UKF::InitializeWeights() {
  weights_ = VectorXd(n_w_);
  weights_(0) = lambda_ / (lambda_ + n_aug_);

  for(int i = 1; i < weights_.size(); i++) {
      weights_(i) = 0.5 / (n_aug_ + lambda_);
    }
}

void UKF::InitializeLaser(const MeasurementPackage &meas_package) {
  x_ << meas_package.raw_measurements_[0], meas_package.raw_measurements_[1], 0, 0, 0;
}

void UKF::InitializeRadar(const MeasurementPackage &meas_package) {
  double x = meas_package.raw_measurements_[0] * cos(meas_package.raw_measurements_[1]);
  double y = meas_package.raw_measurements_[0] * sin(meas_package.raw_measurements_[1]);
  double vx = meas_package.raw_measurements_[2] * cos(meas_package.raw_measurements_[1]);
  double vy = meas_package.raw_measurements_[2] * sin(meas_package.raw_measurements_[1]);
  double v = sqrt(vx * vx + vy * vy);
  x_ << x, y, v, 0, 0;
}

/**
 * Predicts sigma points, the state, and the state covariance matrix.
 * @param {double} delta_t the change in time (in seconds) between the last
 * measurement and this one.
 */
void UKF::Prediction(double delta_t) {
  VectorXd x_aug = VectorXd(n_aug_);
  MatrixXd P_aug = MatrixXd(n_aug_, n_aug_);
  MatrixXd Xsig_aug = MatrixXd(n_aug_, n_sigma_);
    
  x_aug.head(5) = x_;
  x_aug(5) = 0;
  x_aug(6) = 0;

  P_aug.fill(0.0);
  P_aug.topLeftCorner(5, 5) = P_;
  P_aug(5, 5) = std_a_ * std_a_;
  P_aug(6, 6) = std_yawdd_ * std_yawdd_;

  MatrixXd L = P_aug.llt().matrixL();

  Xsig_aug.col(0)  = x_aug;
  double squared_lambda_n_aug = sqrt(lambda_ + n_aug_);
  for (int i = 0; i < n_aug_; i++) {
    Xsig_aug.col(i + 1) = x_aug + squared_lambda_n_aug * L.col(i);
    Xsig_aug.col(i + 1 + n_aug_) = x_aug - squared_lambda_n_aug * L.col(i);
  }  

  for (int i = 0; i < n_sigma_; i++) {
    double p_x = Xsig_aug(0, i);
    double p_y = Xsig_aug(1, i);
    double v = Xsig_aug(2, i);
    double yaw = Xsig_aug(3, i);
    double yawd = Xsig_aug(4, i);
    double nu_a = Xsig_aug(5, i);
    double nu_yawdd = Xsig_aug(6, i);

    double px_p, py_p;

    if (fabs(yawd) > 10e-3) {
        px_p = p_x + v/yawd * (sin(yaw + yawd * delta_t) - sin(yaw));
        py_p = p_y + v/yawd * (cos(yaw) - cos(yaw + yawd * delta_t));
    } else {
        px_p = p_x + v * delta_t * cos(yaw);
        py_p = p_y + v * delta_t * sin(yaw);
    }

    double v_p = v;
    double yaw_p = yaw + yawd * delta_t;
    double yawd_p = yawd;
    
    px_p = px_p + 0.5 * nu_a * delta_t * delta_t * cos(yaw);
    py_p = py_p + 0.5 * nu_a * delta_t * delta_t * sin(yaw);
    v_p = v_p + nu_a * delta_t;

    yaw_p = yaw_p + 0.5 * nu_yawdd * delta_t * delta_t;
    yawd_p = yawd_p + nu_yawdd * delta_t;

    Xsig_pred_(0, i) = px_p;
    Xsig_pred_(1, i) = py_p;
    Xsig_pred_(2, i) = v_p;
    Xsig_pred_(3, i) = yaw_p;
    Xsig_pred_(4, i) = yawd_p;
  }
  
  x_.fill(0.0);
  for (int i = 0; i < n_sigma_; i++) {
    x_ = x_ + weights_(i) * Xsig_pred_.col(i);
  }

  P_.fill(0.0);
  for (int i = 0; i < n_sigma_; i++) {
    VectorXd x_diff = Xsig_pred_.col(i) - x_;
    NormalizeAngle(x_diff(3));

    P_ = P_ + weights_(i) * x_diff * x_diff.transpose();
  }
}

/**
 * Updates the state and the state covariance matrix using a laser measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateLidar(const MeasurementPackage &meas_package) {
  /**
  TODO:

  Complete this function! Use lidar data to update the belief about the object's
  position. Modify the state vector, x_, and covariance, P_.

  You'll also need to calculate the lidar NIS.
  */
  int n_z = 2;
  MatrixXd Zsig = Xsig_pred_.block(0, 0, n_z, n_sigma_);

  //mean predicted measurement
  VectorXd z_pred = VectorXd(n_z);
  z_pred.fill(0.0);
  for (int i = 0; i < n_sigma_; i++) {
      z_pred = z_pred + weights_(i) * Zsig.col(i);
  }

  //measurement covariance matrix S
  MatrixXd S = MatrixXd(n_z,n_z);
  S.fill(0.0);
  for (int i = 0; i < n_sigma_; i++) {
    VectorXd z_diff = Zsig.col(i) - z_pred;
    NormalizeAngle(z_diff(1));
    S = S + weights_(i) * z_diff * z_diff.transpose();
  }

  //add measurement noise covariance matrix
  MatrixXd R = MatrixXd(n_z, n_z);
  R << std_laspx_ * std_laspx_, 0,
        0, std_laspy_ * std_laspy_;

  S = S + R;
  
  VectorXd z_ = meas_package.raw_measurements_;
  
  //create matrix for cross correlation Tc
  MatrixXd Tc = MatrixXd(n_x_, n_z);
  
  //calculate cross correlation matrix
  Tc.fill(0.0);
  for (int i = 0; i < n_sigma_; i++) {
    VectorXd residual_z = Zsig.col(i) - z_pred;
    NormalizeAngle(residual_z(1));

    VectorXd x_diff = Xsig_pred_.col(i) - x_;
    NormalizeAngle(x_diff(3));

    Tc = Tc + weights_(i) * x_diff * residual_z.transpose();
  }

  MatrixXd K = Tc * S.inverse();

  VectorXd residual = z_ - z_pred;

  NormalizeAngle(residual(1));

  x_ = x_ + K * residual;
  P_ = P_ - K * S * K.transpose();
  
  NIS_laser_ = z_.transpose() * S.inverse() * z_;    
}

/**
 * Updates the state and the state covariance matrix using a radar measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateRadar(const MeasurementPackage &meas_package) {
  /**
  TODO:

  Complete this function! Use radar data to update the belief about the object's
  position. Modify the state vector, x_, and covariance, P_.

  You'll also need to calculate the radar NIS.
  */
  //r, phi, and r_dot
  int measurement_dimension = 3;
  
  //create matrix for sigma points in measurement space
  MatrixXd Zsig = MatrixXd(measurement_dimension, n_sigma_);
  
  for (int i = 0; i < n_sigma_; i++) {
    double p_x = Xsig_pred_(0, i);
    double p_y = Xsig_pred_(1, i);
    double v = Xsig_pred_(2, i);
    double yaw = Xsig_pred_(3, i);

    double v1 = cos(yaw) * v;
    double v2 = sin(yaw) * v;

    Zsig(0, i) = sqrt(p_x * p_x + p_y * p_y);
    Zsig(1, i) = atan2(p_y, p_x);
    Zsig(2, i) = (p_x * v1 + p_y * v2) / sqrt(p_x * p_x + p_y * p_y);
  }

  VectorXd z_pred = VectorXd(measurement_dimension);
  z_pred.fill(0.0);
  for (int i = 0; i < n_sigma_; i++) {
      z_pred = z_pred + weights_(i) * Zsig.col(i);
  }

  MatrixXd S = MatrixXd(measurement_dimension, measurement_dimension);
  S.fill(0.0);
  for (int i = 0; i < n_sigma_; i++) {
    VectorXd residual = Zsig.col(i) - z_pred;
    NormalizeAngle(residual(1));
    S = S + weights_(i) * residual * residual.transpose();
  }

  MatrixXd R = MatrixXd(measurement_dimension, measurement_dimension);
  R << std_radr_ * std_radr_, 0, 0,
    0, std_radphi_ * std_radphi_, 0,
    0, 0,std_radrd_ * std_radrd_;
  S = S + R;

  VectorXd z_ = meas_package.raw_measurements_;
  
  MatrixXd Tc = MatrixXd(n_x_, measurement_dimension);
  
  Tc.fill(0.0);
  for (int i = 0; i < n_sigma_; i++) {
    VectorXd residual = Zsig.col(i) - z_pred;
    NormalizeAngle(residual(1));
    VectorXd x_diff = Xsig_pred_.col(i) - x_;
    NormalizeAngle(x_diff(3));

    Tc = Tc + weights_(i) * x_diff * residual.transpose();
  }

  //Kalman gain K;
  MatrixXd K = Tc * S.inverse();

  //residual
  VectorXd z_diff = z_ - z_pred;

  NormalizeAngle(z_diff(1));

  //update state mean and covariance matrix
  x_ = x_ + K * z_diff;
  P_ = P_ - K * S * K.transpose();
  
  NIS_radar_ = z_.transpose() * S.inverse() * z_;  
}

void UKF::NormalizeAngle(double &angle) {
  while (angle > M_PI) {
    angle -= 2. * M_PI;
  }
  while (angle < - M_PI) {
    angle += 2.* M_PI;
  }
}
