#pragma once
  
namespace knn {

// ----------------------------------------------------------------------------------
//  nbeats - container for N-BEATS model for forecasting
//         = a ModuleList for processing blocks (generic/seasonal/trend)
// ----------------------------------------------------------------------------------
class TORCH_API NBeatsImpl : public torch::nn::ModuleListImpl {
 using ModuleListImpl::ModuleListImpl;
 public:
 void reset() override;
 void pretty_print(std::ostream& s) const override;
 void blockforward(std::shared_ptr<torch::nn::Module>& m,Tensor& x,Tensor& y);
 Tensor forward(Tensor x);
};
TORCH_MODULE(NBeats);

} // namespace knn
