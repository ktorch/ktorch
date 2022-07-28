#pragma once

namespace knn {

// --------------------------------------------------------------------------
// argmap - map from given type to enumeration, e.g. Tensor -> Arg::tensor
// argvector - map parameter pack to vector of enumerations
// forwardresult - given module type & forward args, get return type
// maxargs - determine the maximum no. of tensors for forward calculation
// --------------------------------------------------------------------------
template <typename A> static Arg argmap() {
 if(std::is_same<A,bool>::value)
  return Arg::boolean;
 else if(std::is_same<A,Tensor>::value || std::is_same<A,const Tensor&>::value)
  return Arg::tensor;
 else if(std::is_same<A,Tuple>::value || std::is_same<A,const Tuple&>::value || std::is_same<A,torch::optional<std::tuple<Tensor, Tensor>>>::value)
  return Arg::tuple;
 else if(std::is_same<A,Nested>::value || std::is_same<A,const Nested&>::value)
  return Arg::nested;
 else if(std::is_same<A,TensorVector>::value || std::is_same<A,const TensorVector&>::value)
  return Arg::vector;
 else if(std::is_same<A,TensorDict>::value || std::is_same<A,const TensorDict&>::value)
  return Arg::dict;
 else
  return Arg::undefined;
}

template <typename... A> static Args argvector() {
  Args v; v.reserve(sizeof...(A));
  torch::apply([&v](Arg a) { v.push_back(a); }, argmap<A>()...);
  return v;
}

template <typename M,typename... A> static Arg forwardresult() {
 using r=torch::detail::return_type_of_forward_t<M,A...>;
 return argmap<r>();
}

size_t maxargs(const AnyModule&,const char *);

// -------------------------------------------------------------------
// functions to parse k args and return module-specific error messages
// -------------------------------------------------------------------
S code(K,J,Cast,Setting);
S code(const Pairs&,Cast);

c10::optional<Dtype> otype(S);
c10::optional<Dtype> otype(K,J,Cast,Setting);
c10::optional<Dtype> otype(const Pairs&,Cast);

void mpos(K,Cast,J);
void mpair(Cast,const Pairs&);

S mset(Setting);
Setting mset(S x,Cast c=Cast::undefined);
void msetting(K,Setting,K);

bool mbool(K,J,Cast,Setting);
bool mbool(const Pairs&,Cast);

int64_t int64(K,J,Cast,Setting);
int64_t int64(const Pairs&,Cast);

c10::optional<int64_t> int64n(K,J,Cast,Setting);
c10::optional<int64_t> int64n(const Pairs&,Cast);

double mdouble(K,J,Cast,Setting);
double mdouble(const Pairs&,Cast);

c10::optional<double> optdouble(K,J,Cast,Setting);
c10::optional<double> optdouble(const Pairs&,Cast);

LongVector mlongs(K,J,Cast,Setting);
LongVector mlongs(const Pairs&,Cast);

Tensor ltensor(K,J,Cast,Setting);
Tensor ltensor(const Pairs&,Cast);

Tensor ftensor(K,J,Cast,Setting);
Tensor ftensor(const Pairs&,Cast);

DoubleVector mdoubles(K,J,Cast,Setting);
DoubleVector mdoubles(const Pairs&,Cast);

// -------------------------------------------------------------------------------------------------
// exarray - check positional or name-value args for long(s), return expanding array,  else error
// exoptional - similar to exarray, for optional long(s), return expanding array with nulls
// exdouble - similar to exarray, but for double array
// -------------------------------------------------------------------------------------------------
template<size_t D> ExpandingArray<D> exarray(K a,J i,Cast c,Setting s) {
 K x=kK(a)[i];
 TORCH_CHECK(x->t==-KJ || x->t==KJ, msym(c)," ",mset(s),": expected long(s), given ",kname(x->t));
 TORCH_CHECK(x->t==-KJ || x->n==D,  msym(c)," ",mset(s),": expected scalar or ",D,"-element input, given ",x->n,"-element list");
 if(x->t==-KJ)
  return ExpandingArray<D>(x->j);
 else
  return ExpandingArray<D>(IntArrayRef((int64_t*)kJ(x),x->n));
}

template<size_t D> ExpandingArray<D> exarray(const Pairs& p,Cast c) {
 TORCH_CHECK(p.t==-KJ || p.t==KJ,   msym(c)," ",p.k,": expected long(s), given ",kname(p.t));
 TORCH_CHECK(p.t==-KJ || p.v->n==D, msym(c)," ",p.k,": expected scalar or ",D,"-element input, given ",p.v->n,"-element list");
 if(p.t==-KJ)
  return ExpandingArray<D>(p.j);
 else
  return ExpandingArray<D>(IntArrayRef((int64_t*)kJ(p.v),p.v->n));
}

template<size_t D> Exoptional<D> exoptional(J j) {
 return null(j) ? Exoptional<D>(c10::nullopt) : Exoptional<D>(j);
}

template<size_t D> Exoptional<D> exoptional(K x) {
 auto a=Exoptional<D>(IntArrayRef((int64_t*)kJ(x),x->n));
 for(J i=0;i<x->n;++i) if(null((*a)[i].value())) (*a)[i]=c10::nullopt;
 return a;
}

template<size_t D> Exoptional<D> exoptional(K a,J i,Cast c,Setting s) {
 K x=kK(a)[i];
 TORCH_CHECK(x->t==-KJ || x->t==KJ, msym(c)," ",mset(s),": expected long(s), given ",kname(x->t));
 TORCH_CHECK(x->t==-KJ || x->n==D,  msym(c)," ",mset(s),": expected scalar or ",D,"-element input, given ",x->n,"-element list");
 return x->t == -KJ ? exoptional<D>(x->j) : exoptional<D>(x);
}

template<size_t D> Exoptional<D> exoptional(const Pairs& p,Cast c) {
 TORCH_CHECK(p.t==-KJ || p.t==KJ,   msym(c)," ",p.k,": expected long(s), given ",kname(p.t));
 TORCH_CHECK(p.t==-KJ || p.v->n==D, msym(c)," ",p.k,": expected scalar or ",D,"-element input, given ",p.v->n,"-element list");
 return p.t == -KJ ? exoptional<D>(p.j) : exoptional<D>(p.v);
}

template<size_t D> Exdouble<D> exdouble(K a,J i,Cast c,Setting s) {
 K x=kK(a)[i];
 TORCH_CHECK(x->t==-KF || x->t==KF, msym(c)," ",mset(s),": expected double(s), given ",kname(x->t));
 TORCH_CHECK(x->t==-KF || x->n==D,  msym(c)," ",mset(s),": expected scalar or ",D,"-element input, given ",x->n,"-element list");
 if(x->t==-KF)
  return Exdouble<D>(x->f);
 else
  return Exdouble<D>(torch::ArrayRef<double>(kF(x),x->n));
}

template<size_t D> Exdouble<D> exdouble(const Pairs& p,Cast c) {
 TORCH_CHECK(p.t==-KF || p.t==KF,   msym(c)," ",p.k,": expected double(s), given ",kname(p.t));
 TORCH_CHECK(p.t==-KF || p.v->n==D, msym(c)," ",p.k,": expected scalar or ",D,"-element input, given ",p.v->n,"-element list");
 if(p.t==-KF)
  return Exdouble<D>(p.f);
 else
  return Exdouble<D>(torch::ArrayRef<double>(kF(p.v),p.v->n));
}

} // namespace knn
