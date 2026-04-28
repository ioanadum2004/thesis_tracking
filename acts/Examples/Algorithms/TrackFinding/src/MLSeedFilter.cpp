#include "ActsExamples/TrackFinding/MLSeedFilter.hpp"
#include "ActsExamples/Framework/WhiteBoard.hpp"
#include "Acts/Utilities/Logger.hpp"

#include <nlohmann/json.hpp>
#include <onnxruntime_cxx_api.h>

#include <cmath>
#include <fstream>
#include <stdexcept>
#include <vector>

namespace ActsExamples {

MLSeedFilter::MLSeedFilter(Config cfg, Acts::Logging::Level lvl)      /// constructor, initialize data handles and ONNX session
    : IAlgorithm("MLSeedFilter", lvl),
      m_cfg(std::move(cfg)),
      m_env(ORT_LOGGING_LEVEL_WARNING, "MLSeedFilter"),
      m_session(m_env, m_cfg.modelPath.c_str(), Ort::SessionOptions{}) {
  m_inputTrackParameters.initialize(m_cfg.inputTrackParameters);     /// initialize data handles with collection names from config
  m_outputTrackParameters.initialize(m_cfg.outputTrackParameters);
  m_inputSeeds.initialize(m_cfg.inputSeeds);
  m_outputSeeds.initialize(m_cfg.outputSeeds);
}

ProcessCode MLSeedFilter::execute(const AlgorithmContext& ctx) const {     /// runs once per event, main method, read seeds and track parameters, extract features, run inference, apply threshold, write output
  ACTS_LOCAL_LOGGER(logger());
  
  // 1. Read seeds from whiteboard
  const auto& seeds = m_inputSeeds(ctx.eventStore);                /// seeds from estimatedseeds, u call the function w the file in perfect spacepoints
  const auto& params = m_inputTrackParameters(ctx.eventStore);     /// same here for track parameters, estimatedparameters , u call the function w the file in the main file

  ACTS_INFO("MLSeedFilter: seeds in = " << seeds.size());
  ACTS_DEBUG("MLSeedFilter: seeds in = " << seeds.size());

  if (seeds.empty()) {                                             /// if no seeds, write empty collections and return
    m_outputTrackParameters(ctx.eventStore, TrackParametersContainer{});
    m_outputSeeds(ctx.eventStore, SimSeedContainer{});
    return ProcessCode::SUCCESS;
  }

  // 2. Extract features and scale them
  // const int nFeatures = 12;                    /// number of features expected by the model, must match FEATURE_COLS in mlp_model.py CHANGE if u change the nr of features
  const int nFeatures = 27;
  std::vector<float> inputData;
  inputData.reserve(seeds.size() * nFeatures);

  // for (const auto& p : params) {
  //  auto sparams = p.parameters();             /// [loc0, loc1, phi, theta, qop, time] parameters is the function in trackparameterscontainer that gives u the vector of floats with the track parameters
  //  auto cov    = p.covariance().value();      /// covariance is the function in trackparameterscontainer that gives u the covariance matrix, value() is needed because it returns an optional

    /// loc0 and loc1 are kinda the location in the detector  the location where the track crosses a specific detector surface, expressed in that surface's own local coordinate system
    /// loc0 = transverse impact parameter (how far the track is from the beam axis in the transverse plane)
    /// loc1 = longitudinal impact parameter (how far along the beam axis the track appears to come from)

  for (std::size_t i = 0; i < seeds.size(); ++i) {

    const auto& p       = params[i];
    auto sparams        = p.parameters();
    auto cov            = p.covariance().value();

    // ── Track parameter features ──
    
    float loc0  = sparams[Acts::eBoundLoc0];   /// loc0 is the local position along the first measurement direction, it represents the position of the track in the plane perpendicular to the track direction at the point of closest approach to the beamline. It is used in track reconstruction to determine how well the track fits the measurements and to calculate residuals.
    float loc1  = sparams[Acts::eBoundLoc1];   /// loc1 is the local position along the second measurement direction, it represents the position of the track in the plane perpendicular to the track direction at the point of closest approach to the beamline, but in a different direction than loc0. It is used together with loc0 to determine the position of the track in the transverse plane and to calculate residuals.
    float phi   = sparams[Acts::eBoundPhi];    /// phi is the azimuthal angle, angle in the transverse plane, measured from the x-axis, ranges from -pi to pi
    float theta = sparams[Acts::eBoundTheta];  /// theta is the polar angle, angle from the z-axis, ranges from 0 to pi
    float qop   = sparams[Acts::eBoundQOverP]; /// qop is the charge over momentum, q/p, where q is the charge of the particle and p is its momentum. It is used instead of p because it can represent both the magnitude and the sign (charge) of the momentum. For example, a positive qop means a positively charged particle, while a negative qop means a negatively charged particle.

    float pt  = std::sin(theta) / std::abs(qop);
    float eta = -std::log(std::tan(theta / 2.0f));      ///eta is not directly in the track parameters, but can be calculated from theta with the formula eta = -ln(tan(theta/2))

    float err_loc0  = std::sqrt(cov(Acts::eBoundLoc0,  Acts::eBoundLoc0)); /// sqrt(cov(i,i)) gives you the uncertainty (standard deviation) of that parameter. 
    float err_loc1  = std::sqrt(cov(Acts::eBoundLoc1,  Acts::eBoundLoc1));
    float err_phi   = std::sqrt(cov(Acts::eBoundPhi,   Acts::eBoundPhi));
    float err_theta = std::sqrt(cov(Acts::eBoundTheta, Acts::eBoundTheta));
    float err_qop   = std::sqrt(cov(Acts::eBoundQOverP,Acts::eBoundQOverP));

    seedPt.push_back(pt);

    // ── Spacepoint coordinate features ────

    const auto& sp = seeds[i].sp();   // [bottom, middle, top]
    float bX = sp[0]->x(), bY = sp[0]->y(), bZ = sp[0]->z();
    float mX = sp[1]->x(), mY = sp[1]->y(), mZ = sp[1]->z();
    float tX = sp[2]->x(), tY = sp[2]->y(), tZ = sp[2]->z();

    // ── Engineered features ───

    float pull_loc0  = loc0 / (err_loc0 + 1e-9f);
    float pull_loc1  = loc1 / (err_loc1 + 1e-9f);
    float dist_bm    = std::sqrt((mX-bX)*(mX-bX) + (mY-bY)*(mY-bY) + (mZ-bZ)*(mZ-bZ));
    float dist_mt    = std::sqrt((tX-mX)*(tX-mX) + (tY-mY)*(tY-mY) + (tZ-mZ)*(tZ-mZ));
    float dist_bt    = std::sqrt((tX-bX)*(tX-bX) + (tY-bY)*(tY-bY) + (tZ-bZ)*(tZ-bZ));
    float dist_ratio = dist_bm / (dist_mt + 1e-6f);

    
    
    // Scale using scaler sparams from Config
    // Order must match FEATURE_COLS in tree_model.py
    std::vector<float> row = {
        pt, eta, phi, theta, qop, loc0, loc1,
        err_loc0, err_loc1, err_phi, err_theta, err_qop,
	bX, bY, bZ, mX, mY, mZ, tX, tY, tZ,
        pull_loc0, pull_loc1, dist_bm, dist_mt, dist_bt, dist_ratio
    };
    
    for (int i = 0; i < nFeatures; ++i) {
      inputData.push_back((row[i] - m_cfg.scalerMeans[i]) / m_cfg.scalerStds[i]);
    } /// loop over the features, scale them using the means and stds from config, and add to inputData vector which will be used for ONNX inference
  }

  // 3. Run ONNX inference - meaning create input tensor, run session, get output scores adica the probability that each seed is good according to the model
  Ort::MemoryInfo memInfo = Ort::MemoryInfo::CreateCpu(
      OrtArenaAllocator, OrtMemTypeDefault);

  std::vector<int64_t> inputShape  = {static_cast<int64_t>(seeds.size()), nFeatures};  // input shape is (number of seeds, number of features), want to run inference on all seeds at once, and each seed has nFeatures features. The model expects a 2D input where each row corresponds to a seed and each column corresponds to a feature.
  std::vector<int64_t> outputShape = {static_cast<int64_t>(seeds.size()), 1};

  auto inputTensor = Ort::Value::CreateTensor<float>(
      memInfo, inputData.data(), inputData.size(),
      inputShape.data(), inputShape.size());

  const char* inputNames[]  = {"input"};
  const char* outputNames[] = {"output"};

  auto outputTensors = m_session.Run(
      Ort::RunOptions{nullptr},
      inputNames, &inputTensor, 1,
      outputNames, 1);

  float* scores = outputTensors[0].GetTensorMutableData<float>();

  // dynamic threshold

  // 4. Filter seeds above threshold 
  TrackParametersContainer filteredParams;   /// create new containers for filtered seeds and their track parameters, to fill them in the loop and then write them to the whiteboard
  SimSeedContainer filteredSeeds;

  for (std::size_t i = 0; i < params.size(); ++i) {
    float pt        = seedPt[i];
    float threshold = (pt < 0.15f) ? 0.20f : (pt < 0.20f) ? 0.30f : 0.40f;

    
    if (scores[i] >= m_cfg.threshold) {
          filteredParams.push_back(params[i]);   /// keep track parameters
          filteredSeeds.push_back(seeds[i]);     /// keep corresponding seed
      }
  }

  ACTS_DEBUG("MLSeedFilter: seeds out = " << filteredParams.size());
  ACTS_INFO("MLSeedFilter: seeds out = " << filteredParams.size());    /// doesnt work for some reason

  // 5. Write filtered seeds back to whiteboard
  m_outputTrackParameters(ctx.eventStore, std::move(filteredParams));
  m_outputSeeds(ctx.eventStore, std::move(filteredSeeds));

  return ProcessCode::SUCCESS;
}

}  // namespace ActsExamples
