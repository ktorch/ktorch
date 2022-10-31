#include "../ktorch.h"
#include "recur.h"
  
namespace knn {

// --------------------------------------------------------------------------------
// Recur - receive input & hidden state for rnn layer
//         sequential modules in/out apply transformations to input & rnn output
// --------------------------------------------------------------------------------
RecurImpl::RecurImpl(const RecurOptions& o) : options(o) {reset();}

void RecurImpl::reset() {}
void RecurImpl::pretty_print(std::ostream& s) const {s << "knn::Recur";}

void RecurImpl::push_back(const std::shared_ptr<Module>& m) {
 bool e=lstm.is_empty() && gru.is_empty() && rnn.is_empty();
 if(const auto& a=std::dynamic_pointer_cast<torch::nn::SequentialImpl>(m)) {
   if(e) {
    TORCH_CHECK(in.is_empty(), "recur: cannot add a sequential module for output processing until a recurrent module(lstm,gru,rnn) defined");
    in=register_module("in", torch::nn::Sequential(a));
   } else {
    TORCH_CHECK(out.is_empty(), "recur: sequential module for output processing already added, cannot add another sequential module");
    out=register_module("out", torch::nn::Sequential(a));
   }
 } else if(const auto& a=std::dynamic_pointer_cast<torch::nn::LSTMImpl>(m)) {
  TORCH_CHECK(e, "recur: cannot add lstm module, ",(gru.is_empty() ? "rnn" : "gru")," module already defined");
  lstm=register_module("lstm",torch::nn::LSTM(a));
 } else if(const auto& a=std::dynamic_pointer_cast<torch::nn::GRUImpl>(m)) {
  TORCH_CHECK(e, "recur: cannot add gru module, ",(rnn.is_empty() ? "lstm" : "rnn")," module already defined");
  gru=register_module("gru",torch::nn::GRU(a));
 } else if(const auto& a=std::dynamic_pointer_cast<torch::nn::RNNImpl>(m)) {
  TORCH_CHECK(e, "recur: cannot add rnn module, ",(gru.is_empty() ? "lstm" : "gru")," module already defined");
  rnn=register_module("rnn",torch::nn::RNN(a));
 } else {
  TORCH_CHECK(false, 
              "recur: unable to add ",mlabel(m),
              ", expecting sequential modules for input/output processing or recurrent module(lstm,gru,rnn)");
 }
}

std::vector<Tensor> RecurImpl::forward(const Tensor& x, const Tensor& y, const Tensor& z) {
 using OptTuple=c10::optional<Tuple>;
 std::vector<Tensor> v;
 bool i=in.is_empty()  || in->is_empty();   // true if Sequential for input undefined or empty
 bool o=out.is_empty() || out->is_empty();  // true if Sequential for output undefined or empty
 if(!lstm.is_empty()) {
   Nested r=z.defined() ? lstm->forward(i ? x : in->forward(x), OptTuple(std::make_tuple(y,z)))
                        : lstm->forward(i ? x : in->forward(x));
   Tuple& h=std::get<1>(r);
   if(options.detach()) std::get<0>(h).detach_(), std::get<1>(h).detach_();
   v.push_back(o ? std::get<0>(r) : out->forward(std::get<0>(r)));
   v.push_back(std::get<0>(h));
   v.push_back(std::get<1>(h));
 } else if(!gru.is_empty() || !rnn.is_empty()) {
   Tuple r=rnn.is_empty() ? gru->forward(i ? x : in->forward(x), y)
                          : rnn->forward(i ? x : in->forward(x), y);
   if(options.detach()) std::get<1>(r).detach_();
   v.push_back(o ? std::get<0>(r) : out->forward(std::get<0>(r)));
   v.push_back(std::get<1>(r));
 } else {
  TORCH_CHECK(false, "recur: no recurrent lstm, gru or rnn module defined");
 }
 return v;
}

// ---------------------------------------------------------------------------
// recur - get/set options (currently only true/false flag for detach)
// ---------------------------------------------------------------------------
RecurOptions recur(K x,J i,Cast c) {
 RecurOptions o; Pairs p; J n=xargc(x,i,p);
 for(J j=0;j<n;++j)
  switch(j) {
   case 0: o.detach(mbool(x,i+j,c,Setting::detach)); break;
   default: TORCH_ERROR(msym(c),": 1 positional arg(detach flag) expected, ",n," supplied");
  }
 while(xpair(p))
  switch(mset(p.k,c)) {
   case Setting::detach: o.detach(mbool(p,c)); break;
   default: mpair(c,p); break;
  }
 return o;
}

K recur(bool a,const RecurOptions& o) {
 K x=KDICT;
 if(a || o.detach() != RecurOptions().detach()) msetting(x, Setting::detach, kb(o.detach()));
 return resolvedict(x);
}

// --------------------------------------------------------------------------------------
// rnn - create rnn/gru/lstm module given options/set dictionary of options from module
//     - rnn accepts non-linear function specification: `tanh or `relu
//       gru & lstm don't have that option, so templates/overloading used for fn setting
// --------------------------------------------------------------------------------------
void rnnfn(torch::nn::RNNOptions& o,Cast c,S s) {
 switch(emap(s)) {
  case Enum::tanh:   o.nonlinearity(torch::kTanh); break;
  case Enum::relu:   o.nonlinearity(torch::kReLU); break;
  default: TORCH_ERROR("unrecognized RNN fn: ",s); break;
 }
}

torch::nn::RNNOptions rnn(K x,J i,Cast c) {
 torch::nn::RNNOptions o(0,0); Pairs p; Tensor w; J n=xargc(x,i,p);
 for(J j=0;j<n;++j)
  switch(j) {
   case 0: o.input_size (int64(x,i+j,c,Setting::in)); break;
   case 1: o.hidden_size(int64(x,i+j,c,Setting::hidden)); break;
   case 2: o.num_layers (int64(x,i+j,c,Setting::layers)); break;
   case 3: rnnfn(o,c,code(x,i+j,c,Setting::fn)); break;
   case 4: o.bias(mbool(x,i+j,c,Setting::bias)); break;
   case 5: o.batch_first(mbool(x,i+j,c,Setting::batchfirst)); break;
   case 6: o.dropout(mdouble(x,i+j,c,Setting::dropout)); break;
   case 7: o.bidirectional(mbool(x,i+j,c,Setting::bi)); break;
   default: mpos(x,c,i+j); break;
  }
 rnnpair(c,p,o);
 return o;
}

S rnnfn(const torch::nn::RNNOptions& o) {return ESYM(o.nonlinearity());}

}  // knn namespace
