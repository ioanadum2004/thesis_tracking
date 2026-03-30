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

MLSeedFilter::MLSeedFilter(Config cfg, Acts::Logging::Level lvl)
    : IAlgorithm("MLSeedFilter", lvl),
      m_cfg(std::move(cfg)),
      m_env(ORT_LOGGING_LEVEL_WARNING, "MLSeedFilter"),
      m_session(m_env, m_cfg.modelPath.c_str(), Ort::SessionOptions{}) {
  m_inputTrackParameters.initialize(m_cfg.inputTrackParameters);
  m_outputTrackParameters.initialize(m_cfg.outputTrackParameters);
  m_inputSeeds.initialize(m_cfg.inputSeeds);
  m_outputSeeds.initialize(m_cfg.outputSeeds);
}

ProcessCode MLSeedFilter::execute(const AlgorithmContext& ctx) const {
  // 1. Read seeds from whiteboard
  const auto& seeds = m_inputSeeds(ctx.eventStore);
  const auto& params = m_inputTrackParameters(ctx.eventStore);

  ACTS_INFO("MLSeedFilter: seeds in = " << seeds.size());
  ACTS_DEBUG("MLSeedFilter: seeds in = " << seeds.size());

  if (seeds.empty()) {
    m_outputTrackParameters(ctx.eventStore, TrackParametersContainer{});
    m_outputSeeds(ctx.eventStore, SimSeedContainer{});
    return ProcessCode::SUCCESS;
  }

  // 2. Extract features and scale them
  const int nFeatures = 12;
  std::vector<float> inputData;
  inputData.reserve(seeds.size() * nFeatures);

  //for (const auto& seed : seeds) {
  for (const auto& p : params) {
    auto sparams = p.parameters();
    auto cov    = p.covariance().value();

    float loc0  = sparams[Acts::eBoundLoc0];
    float loc1  = sparams[Acts::eBoundLoc1];
    float phi   = sparams[Acts::eBoundPhi];
    float theta = sparams[Acts::eBoundTheta];
    float qop   = sparams[Acts::eBoundQOverP];

    float pt  = std::sin(theta) / std::abs(qop);
    float eta = -std::log(std::tan(theta / 2.0f));

    float err_loc0  = std::sqrt(cov(Acts::eBoundLoc0,  Acts::eBoundLoc0));
    float err_loc1  = std::sqrt(cov(Acts::eBoundLoc1,  Acts::eBoundLoc1));
    float err_phi   = std::sqrt(cov(Acts::eBoundPhi,   Acts::eBoundPhi));
    float err_theta = std::sqrt(cov(Acts::eBoundTheta, Acts::eBoundTheta));
    float err_qop   = std::sqrt(cov(Acts::eBoundQOverP,Acts::eBoundQOverP));

    // Scale using scaler sparams from Config
    // Order must match FEATURE_COLS in tree_model.py
    std::vector<float> row = {
        pt, eta, phi, theta, qop, loc0, loc1,
        err_loc0, err_loc1, err_phi, err_theta, err_qop
    };
    for (int i = 0; i < nFeatures; ++i) {
      inputData.push_back((row[i] - m_cfg.scalerMeans[i]) / m_cfg.scalerStds[i]);
    }
  }

  // 3. Run ONNX inference
  Ort::MemoryInfo memInfo = Ort::MemoryInfo::CreateCpu(
      OrtArenaAllocator, OrtMemTypeDefault);

  std::vector<int64_t> inputShape  = {static_cast<int64_t>(seeds.size()), nFeatures};
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

  // 4. Filter seeds above threshold
  TrackParametersContainer filteredParams;
  SimSeedContainer filteredSeeds;

  for (std::size_t i = 0; i < params.size(); ++i) {
      if (scores[i] >= m_cfg.threshold) {
          filteredParams.push_back(params[i]);
          filteredSeeds.push_back(seeds[i]);
      }
  }

  ACTS_DEBUG("MLSeedFilter: seeds out = " << filteredParams.size());
  ACTS_INFO("MLSeedFilter: seeds out = " << filteredParams.size());

  // 5. Write filtered seeds back to whiteboard
  m_outputTrackParameters(ctx.eventStore, std::move(filteredParams));
  m_outputSeeds(ctx.eventStore, std::move(filteredSeeds));

  return ProcessCode::SUCCESS;
}

}  // namespace ActsExamples