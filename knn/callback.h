#pragma once
#include "util.h"

namespace knn {

// ------------------------------------------------------------------------
// cbinput: given Pytorch tensor(s), return k container to pass to callback
// ------------------------------------------------------------------------
K cbinput(const Tensor&);
K cbinput(const Tuple&);
K cbinput(const Nested&);
K cbinput(const TensorVector&);

// ------------------------------------------------------------------------
// cbfree: free array of K values (result, args) from callback
// cbresult: get tensor(s) from k result returned from callback
// ------------------------------------------------------------------------
template<size_t N> void cbfree(std::array<K,N> a) {
 for(auto x:a)
  if(x) xfree(x), r0(x);
}

bool cbresult(K,Tensor&);
bool cbresult(K,Tuple&);
bool cbresult(K,Nested&);
bool cbresult(K,TensorVector&);

template<typename R,size_t N> R cbresult(const std::array<K,N>& a) {
 R r; K z=a[0];
 if(!z) {
  cbfree(a);
  TORCH_ERROR("callback: unexpected null return from k function");
 } else if(z->t == -128) {
  std::string s(z->s);   // retrieve error before de-reference
  cbfree(a);             // free error result & arg(s)
  TORCH_ERROR("callback error from k: ",s);
 } else if(!cbresult(z,r)) {
  std::string s(kname(z));
  cbfree(a);
  TORCH_ERROR("callback: unable to retrieve ",argname(argmap<R>())," from ",s);
 }
 cbfree(a);
 return r;
}

// ------------------------------------------------------------------------
// cbget: get i'th argument from variable list
// ------------------------------------------------------------------------
/*
template <int I, typename... X> decltype(auto) cbget(X&&... x) {
  return std::get<I>(std::forward_as_tuple(x...));
}
*/

template<size_t I,typename T,typename... X,typename std::enable_if<I==0>::type* = nullptr>
inline constexpr decltype(auto) cbget(T&& t, X&&... x) {
    return std::forward<T>(t);
}

template<size_t I, typename T, typename... X, typename std::enable_if<(I>0 && I<=sizeof...(X))>::type* = nullptr>
inline constexpr decltype(auto) cbget(T&& t, X&&... x) {
    return cbget<I-1>(std::forward<X>(x)...);
}

// -------------------------------------------------------------------------------
// cbforward: forward with 1-3 inputs, calls k function, frees intermediate args
// clonedict: used in Callback clone method to clone parameters and buffers
// -------------------------------------------------------------------------------
template <typename... A,size_t N=sizeof...(A),typename std::enable_if_t<N==1,int> = 1>
std::array<K,N+1> cbforward(K m,const std::string& f,A...a) {
 K x=cbinput(cbget<0>(a...));
 K r=k(0,(S)f.c_str(),r1(m),r1(x),(K)0);
 return {{r,x}};
}

template <typename... A,size_t N=sizeof...(A),typename std::enable_if_t<N==2,int> = 1>
std::array<K,N+1> cbforward(K m,const std::string& f,A...a) {
 K x=cbinput(cbget<0>(a...)), y=cbinput(cbget<1>(a...));
 K r=k(0,(S)f.c_str(),r1(m),r1(x),r1(y),(K)0);
 return {{r,x,y}};
}

template <typename... A,size_t N=sizeof...(A),typename std::enable_if_t<N==3,int> = 1>
std::array<K,N+1> cbforward(K m,const std::string& f,A...a) {
 K x=cbinput(cbget<0>(a...)), y=cbinput(cbget<1>(a...)), z=cbinput(cbget<2>(a...));
 K r=k(0,(S)f.c_str(),r1(m),r1(x),r1(y),r1(z),(K)0);
 return {{r,x,y,z}};
}

void clonedict(const TensorDict&,TensorDict&,const c10::optional<Device>& d);
 
// ---------------------------------------------------------------------------
// callback options -- function name/definition, input & output types
// ---------------------------------------------------------------------------
struct TORCH_API CallbackOptions {
 CallbackOptions() = default;
 CallbackOptions(std::string f) : fn_(std::move(f)) {}
 TORCH_ARG(std::string, fn);                  //function name or string, e.g. `f or "{[m;x;y] mul(x;y)}"
 TORCH_ARG(bool, fnstring);                   //true if fn given as string(or expression), false if sym
 TORCH_ARG(Args, in) = Args{Arg::tensor};     //type of input(s)
 TORCH_ARG(Arg, out) = Arg::tensor;           //type of output
};

// ---------------------------------------------------------------------------
// callback base template
// ---------------------------------------------------------------------------
template<typename T>class TORCH_API CallbackBase : public torch::nn::Cloneable<T> {
 public:
 explicit CallbackBase(const CallbackOptions& o) : options(o) {reset();}

 void reset() override {};

 void pretty_print(std::ostream& s) const override {
  s << "knn::Callback(fn=" << options.fn() << ", in=";
  for(auto a:options.in()) s << argname(a) << ",";
  s << " out=" << argname(options.out()) << ")";
 }

 template<typename R=Tensor,typename... X> R forward(X&&... in) {
  TORCH_CHECK(options.fn().size(), "callback: no k callback function defined");
  K m=kmodule(Cast::callback,this->shared_from_this());
  auto a=cbforward(m,options.fn(),in...); // array of K values: result, args..
  kfree(m); r0(m);
  return cbresult<R>(a);
 }

 virtual AnyModule any() {TORCH_ERROR("unable to make type-erased module");}

 template<typename A> Moduleptr clone(const c10::optional<Device>& d=c10::nullopt) const {
  torch::NoGradGuard g;
  const auto& a=static_cast<const A&>(*this);
  auto b=std::make_shared<A>(a);
  clonedict(this->named_parameters(false), b->parameters_, d);
  clonedict(this->named_buffers(false), access_private::buffers_(*b), d);
  for(const auto& m:this->named_children())
   access_private::children_(*b)[m.key()]=std::move(m.value()->clone(d));
  return b;
 }

 CallbackOptions options;
};

class TORCH_API CallbackImpl : public CallbackBase<CallbackImpl> {
 public:
 using CallbackBase::CallbackBase;
};
TORCH_MODULE(Callback);

// ---------------------------------------------------------------------------
//  callbacks with tensor result and 1-3 input tensors
// ---------------------------------------------------------------------------
class TORCH_API TensorToTensorImpl : public CallbackImpl {
 public:
 using CallbackImpl::CallbackImpl;
 AnyModule any() override;
 Moduleptr clone(const c10::optional<Device>& d=c10::nullopt) const override;
 Tensor forward(const Tensor& x);
};
TORCH_MODULE(TensorToTensor);

class TORCH_API Tensor2ToTensorImpl : public CallbackImpl {
 public:
 using CallbackImpl::CallbackImpl;
 AnyModule any() override;
 Moduleptr clone(const c10::optional<Device>& d=c10::nullopt) const override;
 Tensor forward(const Tensor& x,const Tensor& y);
};
TORCH_MODULE(Tensor2ToTensor);

class TORCH_API Tensor3ToTensorImpl : public CallbackImpl {
 public:
 using CallbackImpl::CallbackImpl;
 AnyModule any() override;
 Moduleptr clone(const c10::optional<Device>& d=c10::nullopt) const override;
 Tensor forward(const Tensor& x,const Tensor& y,const Tensor& z);
};
TORCH_MODULE(Tensor3ToTensor);

// ----------------------------------------------------------------------
// support functions for callback modules: setting & retrieving options
// ----------------------------------------------------------------------
K cbfn(const CallbackOptions& o);
Moduleptr callback(K,J,Cast);
K callback(bool,bool,const CallbackImpl&);
Callbacks callbacks();

} // namespace knn
