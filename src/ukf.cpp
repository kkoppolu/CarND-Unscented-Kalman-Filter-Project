#include "ukf.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

namespace
{
double normalizeAngle(double angle)
{
  while (angle > M_PI)
  {
    angle -= 2 * M_PI;
  }
  while (angle < -M_PI)
  {
    angle += 2 * M_PI;
  }
  return angle;
}
}

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF()
    : is_initialized_(false), use_laser_(true), use_radar_(true), n_x_(5), n_aug_(n_x_ + 2), lambda_(3 - n_x_), time_us_(0), std_a_(1), std_yawdd_(6), std_laspx_(0.15), std_laspy_(0.15), std_radr_(0.3), std_radphi_(0.03), std_radrd_(0.3)
{
  weights_ = VectorXd(2 * n_aug_ + 1);

  // set weights
  double weight_0 = lambda_ / (lambda_ + n_aug_);
  weights_(0) = weight_0;
  for (int i = 1; i < 2 * n_aug_ + 1; i++)
  { //2n+1 weights
    double weight = 0.5 / (n_aug_ + lambda_);
    weights_(i) = weight;
  }

  // initial state
  x_ = VectorXd(n_x_);
  x_ << 0, 0, 0, 0, 0;
  // initial co-variance
  P_ = MatrixXd::Identity(n_x_, n_x_);

  Xsig_pred_ = MatrixXd(n_x_, (2 * n_aug_) + 1);
  Xsig_pred_.fill(0.0);
  std::cout << "UKF construction complete" << std::endl;
}

UKF::~UKF() {}

void UKF::initFilter(const MeasurementPackage &measurement)
{
  if (measurement.sensor_type_ == MeasurementPackage::LASER)
  {
    // record the initial x and y position. we do not have velocity information. SO, assume 0
    x_ << measurement.raw_measurements_[0], measurement.raw_measurements_[1], 0, 0, 0;
  }
  else
  { // if (sensorType == MeasurementPackage::RADAR
    // derive the initial x and y positions from the polar co-ordinates
    const float rho = measurement.raw_measurements_[0];
    float phi = measurement.raw_measurements_[1];
    phi = normalizeAngle(phi);

    const double px = rho * cos(phi);
    const double py = rho * sin(phi);

    double si = 0.5 * M_PI - phi;
    si = normalizeAngle(si);
    x_ << px, py, rho, si, 0;
  }

  is_initialized_ = true;
  std::cout << "Filter initialization complete" << std::endl;
}

/**
 * @param {MeasurementPackage} meas_package The latest measurement data of
 * either radar or laser.
 */
void UKF::ProcessMeasurement(MeasurementPackage meas_package)
{
  // delt in secs
  const double delt = (meas_package.timestamp_ - time_us_) / 1000000.0;
  time_us_ = meas_package.timestamp_;
  if (!is_initialized_)
  {
    initFilter(meas_package);
    return;
  }

  if (meas_package.sensor_type_ == MeasurementPackage::LASER && use_laser_)
  {
    Prediction(delt);
    UpdateLidar(meas_package);
  }
  else if (use_radar_)
  {
    Prediction(delt);
    UpdateRadar(meas_package);
  }
}

/**
 * Predicts sigma points, the state, and the state covariance matrix.
 * @param {double} delta_t the change in time (in seconds) between the last
 * measurement and this one.
 */
void UKF::Prediction(double delta_t)
{
  std::cout << "Prediction start" << std::endl;
  // generate sigma points
  MatrixXd Xsig_aug;
  augmentSigmaPoints(Xsig_aug);

  // predict sigma points
  sigmaPointPrediction(Xsig_aug, delta_t);

  // estimate mean and co-variance
  predictMeanAndCovariance();
}

void UKF::augmentSigmaPoints(MatrixXd &Xsig_aug)
{
  //create augmented mean vector
  VectorXd x_aug = VectorXd(n_aug_);
  //create augmented state covariance
  MatrixXd P_aug = MatrixXd(n_aug_, n_aug_);

  //create augmented mean state
  x_aug.head(n_x_) = x_;
  x_aug(n_x_) = 0;
  x_aug(n_x_ + 1) = 0;
  //create augmented covariance matrix
  P_aug.fill(0.0);
  P_aug.topLeftCorner(n_x_, n_x_) = P_;
  P_aug(n_x_, n_x_) = std_a_ * std_a_;
  P_aug(n_x_ + 1, n_x_ + 1) = std_yawdd_ * std_yawdd_;

  //create sigma point matrix
  Xsig_aug = MatrixXd(n_aug_, 2 * n_aug_ + 1);

  //create square root matrix
  MatrixXd L = P_aug.llt().matrixL();

  //create augmented sigma points
  Xsig_aug.col(0) = x_aug;
  for (int i = 0; i < n_aug_; i++)
  {
    Xsig_aug.col(i + 1) = x_aug + sqrt(lambda_ + n_aug_) * L.col(i);
    Xsig_aug.col(i + 1 + n_aug_) = x_aug - sqrt(lambda_ + n_aug_) * L.col(i);
  }
}

void UKF::sigmaPointPrediction(const MatrixXd &Xsig_aug, double delta_t)
{

  //predict sigma points
  for (int i = 0; i < 2 * n_aug_ + 1; i++)
  {
    //extract values for better readability
    double p_x = Xsig_aug(0, i);
    double p_y = Xsig_aug(1, i);
    double v = Xsig_aug(2, i);
    double yaw = Xsig_aug(3, i);
    double yawd = Xsig_aug(4, i);
    double nu_a = Xsig_aug(5, i);
    double nu_yawdd = Xsig_aug(6, i);

    //predicted state values
    double px_p, py_p;

    //avoid division by zero
    if (fabs(yawd) > 0.001)
    {
      px_p = p_x + v / yawd * (sin(yaw + yawd * delta_t) - sin(yaw));
      py_p = p_y + v / yawd * (cos(yaw) - cos(yaw + yawd * delta_t));
    }
    else
    {
      px_p = p_x + v * delta_t * cos(yaw);
      py_p = p_y + v * delta_t * sin(yaw);
    }

    double v_p = v;
    double yaw_p = yaw + yawd * delta_t;
    double yawd_p = yawd;

    //add noise
    px_p = px_p + 0.5 * nu_a * delta_t * delta_t * cos(yaw);
    py_p = py_p + 0.5 * nu_a * delta_t * delta_t * sin(yaw);
    v_p = v_p + nu_a * delta_t;

    yaw_p = yaw_p + 0.5 * nu_yawdd * delta_t * delta_t;
    yawd_p = yawd_p + nu_yawdd * delta_t;

    //write predicted sigma point into right column
    Xsig_pred_(0, i) = px_p;
    Xsig_pred_(1, i) = py_p;
    Xsig_pred_(2, i) = v_p;
    Xsig_pred_(3, i) = yaw_p;
    Xsig_pred_(4, i) = yawd_p;
  }
}

void UKF::predictMeanAndCovariance()
{
  std::cout << "Prediction mean and co-variance start" << std::endl;
  //predicted state mean
  x_.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; i++)
  {
    //iterate over sigma points
    x_ = x_ + weights_(i) * Xsig_pred_.col(i);
  }
  x_(3) = ::normalizeAngle(x_(3));

  //predicted state covariance matrix
  P_.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; i++)
  {
    //iterate over sigma points
    // state difference
    VectorXd x_diff = Xsig_pred_.col(i) - x_;
    x_diff(3) = ::normalizeAngle(x_diff(3));

    P_ = P_ + weights_(i) * x_diff * x_diff.transpose();
  }
}

/**
 * Updates the state and the state covariance matrix using a laser measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateLidar(MeasurementPackage meas_package)
{
  std::cout << "update LIDAR start" << std::endl;
  //set measurement dimension, LIDAR can meqasure px and py
  int n_z = 2;
  VectorXd z_pred(n_z);
  MatrixXd S = MatrixXd(n_z, n_z);
  //create matrix for sigma points in measurement space
  MatrixXd Zsig = MatrixXd(n_z, 2 * n_aug_ + 1);
  predictLidarSigmaPoints(z_pred, S, Zsig);

  // incoming LIDAR measurement
  VectorXd z = VectorXd(n_z);
  z << meas_package.raw_measurements_[0],
      meas_package.raw_measurements_[1];

  //calculate cross correlation matrix
  MatrixXd Tc = MatrixXd(n_x_, n_z); 
  Tc.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; i++)
  { //residual
    VectorXd z_diff = Zsig.col(i) - z_pred;

    // state difference
    VectorXd x_diff = Xsig_pred_.col(i) - x_;
    x_diff(3) = ::normalizeAngle(x_diff(3));

    Tc = Tc + weights_(i) * x_diff * z_diff.transpose();
  }

  //Kalman gain K;
  MatrixXd K = Tc * S.inverse();

  //residual
  VectorXd z_diff = z - z_pred;
  
  //update state mean and covariance matrix
  x_ = x_ + K * z_diff;
  P_ = P_ - K * S * K.transpose();

  nis_lidar_ = z_diff.transpose() * S.inverse() * z_diff;
  /**
  TODO:

  Complete this function! Use lidar data to update the belief about the object's
  position. Modify the state vector, x_, and covariance, P_.

  You'll also need to calculate the lidar NIS.
  */
}

void UKF::predictLidarSigmaPoints(VectorXd &z_pred, MatrixXd &S, MatrixXd& Zsig)
{
  //transform sigma points into measurement space
  for (int i = 0; i < 2 * n_aug_ + 1; i++)
  { 
    double p_x = Xsig_pred_(0, i);
    double p_y = Xsig_pred_(1, i);

    // measurement model
    Zsig(0, i) = p_x;
    Zsig(1, i) = p_y;
  }

  //mean predicted measurement
  z_pred.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; i++)
  {
    z_pred = z_pred + weights_(i) * Zsig.col(i);
  }

  //measurement covariance matrix S
  S.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; i++)
  { 
    //residual
    VectorXd z_diff = Zsig.col(i) - z_pred;
    S = S + weights_(i) * z_diff * z_diff.transpose();
  }

  //add measurement noise covariance matrix
  MatrixXd R = MatrixXd(z_pred.rows(), z_pred.rows());
  R << std_laspx_ * std_laspx_, 0,
      0, std_laspy_ * std_laspy_;
  S = S + R;
}

void UKF::predictRadarSigmaPoints(VectorXd &z_pred, MatrixXd &S, MatrixXd& Zsig)
{
  std::cout << "Predict RADAR points start" << std::endl;
  //transform sigma points into measurement space
  for (int i = 0; i < 2 * n_aug_ + 1; i++)
  { 
    double p_x = Xsig_pred_(0, i);
    double p_y = Xsig_pred_(1, i);
    double v = Xsig_pred_(2, i);
    double yaw = Xsig_pred_(3, i);

    double v1 = cos(yaw) * v;
    double v2 = sin(yaw) * v;

    // measurement model
    Zsig(0, i) = sqrt(p_x * p_x + p_y * p_y);                         //r
    Zsig(1, i) = atan2(p_y, p_x);                                     //phi
    Zsig(2, i) = (p_x * v1 + p_y * v2) / sqrt(p_x * p_x + p_y * p_y); //r_dot
  }

  //mean predicted measurement
  z_pred.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; i++)
  {
    z_pred = z_pred + weights_(i) * Zsig.col(i);
  }

  //measurement covariance matrix S
  S.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; i++)
  { 
    //residual
    VectorXd z_diff = Zsig.col(i) - z_pred;
    z_diff(1) = ::normalizeAngle(z_diff(1));

    S = S + weights_(i) * z_diff * z_diff.transpose();
  }

  //add measurement noise covariance matrix
  MatrixXd R = MatrixXd(z_pred.rows(), z_pred.rows());
  R << std_radr_ * std_radr_, 0, 0,
      0, std_radphi_ * std_radphi_, 0,
      0, 0, std_radrd_ * std_radrd_;
  S = S + R;
}

/**
 * Updates the state and the state covariance matrix using a radar measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateRadar(MeasurementPackage meas_package)
{
  std::cout << "Update RADAR start" << std::endl;
  //set measurement dimension, radar can measure r, phi, and r_dot
  int n_z = 3;
  VectorXd z_pred(n_z);
  MatrixXd S = MatrixXd(n_z, n_z);
  //create matrix for sigma points in measurement space
  MatrixXd Zsig = MatrixXd(n_z, 2 * n_aug_ + 1);
  predictRadarSigmaPoints(z_pred, S, Zsig);

  // incoming radar measurement
  VectorXd z = VectorXd(n_z);
  z << meas_package.raw_measurements_[0],
      meas_package.raw_measurements_[1],
      meas_package.raw_measurements_[2];

  //calculate cross correlation matrix
  MatrixXd Tc = MatrixXd(n_x_, n_z); 
  Tc.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; i++)
  { //residual
    VectorXd z_diff = Zsig.col(i) - z_pred;
    z_diff(1) = ::normalizeAngle(z_diff(1));

    // state difference
    VectorXd x_diff = Xsig_pred_.col(i) - x_;
    x_diff(3) = ::normalizeAngle(x_diff(3));

    Tc = Tc + weights_(i) * x_diff * z_diff.transpose();
  }

  //Kalman gain K;
  MatrixXd K = Tc * S.inverse();

  //residual
  VectorXd z_diff = z - z_pred;
  z_diff(1) = ::normalizeAngle(z_diff(1));

  //update state mean and covariance matrix
  x_ = x_ + K * z_diff;
  P_ = P_ - K * S * K.transpose();

  nis_radar_ = z_diff.transpose() * S.inverse() * z_diff;
  
  /**
  TODO:

  Complete this function! Use radar data to update the belief about the object's
  position. Modify the state vector, x_, and covariance, P_.

  You'll also need to calculate the radar NIS.
  */
}
