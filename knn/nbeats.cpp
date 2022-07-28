#include "../ktorch.h"
#include "nbeats.h"

namespace knn {

// ----------------------------------------------------------------------------------
//  nbeats - container for N-BEATS model for forecasting
//         = a ModuleList for processing blocks (generic/seasonal/trend)
// ----------------------------------------------------------------------------------
void NBeatsImpl::reset() {}

void NBeatsImpl::pretty_print(std::ostream& stream) const {
 stream << "knn::NBeats";
}

void NBeatsImpl::blockforward(std::shared_ptr<torch::nn::Module>& m,Tensor& x,Tensor& y) {
 if(auto *a=m->as<torch::nn::Sequential>()) {
  Tensor b,f; std::tie(b,f)=a->forward<Tuple>(x);
  x=x-b; y=y.defined() ? y+f : f;
 } else if(auto *a=m->as<torch::nn::ModuleList>()) {
  for(auto& q:a->children())
   blockforward(q,x,y);
 } else {
  TORCH_CHECK(false,"nbeats: not a sequential or list module");
 }
}
  
Tensor NBeatsImpl::forward(Tensor x) {
 Tensor y; size_t i=0;
 for(auto& m:children())
  blockforward(m,x,y), i++;
 return y;
}

} // namespace knn
