#pragma once
#include "util.h"

#ifdef __clang__
# pragma clang diagnostic push
# pragma clang diagnostic ignored "-Wgnu-zero-variadic-macro-arguments"    // FORWARD_HAS_DEFAULT_ARGS
#endif
  
namespace knn {

// --------------------------------------------------------------------------------
// Recur - receive input & hidden state for rnn layer
//         sequential modules in/out apply transformations to input & rnn output
// --------------------------------------------------------------------------------
struct TORCH_API RecurOptions {
 RecurOptions(bool d=true) : detach_(d) {}
 TORCH_ARG(bool, detach);
};

class TORCH_API RecurImpl : public torch::nn::Cloneable<RecurImpl> {
 public:
 explicit RecurImpl(const RecurOptions& o);

 void reset() override;
 void pretty_print(std::ostream& s) const override;
 void push_back(const std::shared_ptr<Module>& m);
 std::vector<Tensor> forward(const Tensor& x, const Tensor& y={}, const Tensor& z={});

 FORWARD_HAS_DEFAULT_ARGS({1, torch::nn::AnyValue(torch::Tensor())},
                          {2, torch::nn::AnyValue(torch::Tensor())})

 RecurOptions options;
 torch::nn::Sequential in=nullptr;
 torch::nn::LSTM lstm=nullptr;
 torch::nn::GRU   gru=nullptr;
 torch::nn::RNN   rnn=nullptr;
 torch::nn::Sequential out=nullptr;
};
TORCH_MODULE(Recur);

RecurOptions recur(K,J,Cast);
K recur(bool,const RecurOptions&);

// --------------------------------------------------------------------------------------
// rnn - create rnn/gru/lstm module given options/set dictionary of options from module
//     - rnn accepts non-linear function specification: `tanh or `relu
//       gru & lstm don't have that option, so templates/overloading used for fn setting
// --------------------------------------------------------------------------------------
template<typename O> static void rnnfn(O& o,Cast c,S s) {
 TORCH_ERROR(msym(c),": no non-linear function required (RNN only)");
}

void rnnfn(torch::nn::RNNOptions&,Cast,S);

template<typename O> static void rnnpair(Cast c,Pairs& p,O& o) {
 while(xpair(p))
  switch(mset(p.k,c)) {
   case Setting::in:          o.input_size(int64(p,c)); break;
   case Setting::hidden:      o.hidden_size(int64(p,c)); break;
   case Setting::layers:      o.num_layers(int64(p,c)); break;
   case Setting::fn:          rnnfn(o,c,c==Cast::rnn ? code(p,c) : nullptr); break;
   case Setting::bias:        o.bias(mbool(p,c)); break;
   case Setting::batchfirst:  o.batch_first(mbool(p,c)); break;
   case Setting::dropout:     o.dropout(mdouble(p,c)); break;
   case Setting::bi:          o.bidirectional(mbool(p,c)); break;
   default: mpair(c,p); break;
  }
 TORCH_CHECK(o.hidden_size()>0, msym(c), ": hidden size should be greater than zero");
}

torch::nn::RNNOptions rnn(K,J,Cast);

template<typename O> static O rnn(K x,J i,Cast c) {
 O o(0,0); Pairs p; Tensor w; J n=xargc(x,i,p);
 for(J j=0;j<n;++j)
  switch(j) {
   case 0: o.input_size (int64(x,i+j,c,Setting::in)); break;
   case 1: o.hidden_size(int64(x,i+j,c,Setting::hidden)); break;
   case 2: o.num_layers (int64(x,i+j,c,Setting::layers)); break;
   case 3: o.bias(mbool(x,i+j,c,Setting::bias)); break;
   case 4: o.batch_first(mbool(x,i+j,c,Setting::batchfirst)); break;
   case 5: o.dropout(mdouble(x,i+j,c,Setting::dropout)); break;
   case 6: o.bidirectional(mbool(x,i+j,c,Setting::bi)); break;
   default: mpos(x,c,i+j); break;
  }
 rnnpair(c,p,o);
 return o;
}

S rnnfn(const torch::nn::RNNOptions&);
template<typename O>S rnnfn(const O& o) {return nullptr;}

template<typename O> K rnn(bool a,const O& o) {
 K x=KDICT; O d(o.input_size(),o.hidden_size()); S s=rnnfn(o);
 msetting(x, Setting::in,     kj(o.input_size()));
 msetting(x, Setting::hidden, kj(o.hidden_size()));
 if(a || (o.num_layers()    != d.num_layers()))   msetting(x, Setting::layers,     kj(o.num_layers()));
 if((a && s) || s           != rnnfn(d))          msetting(x, Setting::fn,         ks(s));
 if(a || (o.bias()          != d.bias()))         msetting(x, Setting::bias,       kb(o.bias()));
 if(a || (o.batch_first()   != d.batch_first()))  msetting(x, Setting::batchfirst, kb(o.batch_first()));
 if(a || (o.dropout()       != d.dropout()))      msetting(x, Setting::dropout,    kf(o.dropout()));
 if(a || (o.bidirectional() != d.bidirectional()))msetting(x, Setting::bi,         kb(o.bidirectional()));
 return x;
}

}  // knn namespace

#ifdef __clang__
# pragma clang diagnostic pop
#endif
