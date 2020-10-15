#include "ktorch.h"
namespace nn=torch::nn;
namespace fnn=torch::nn::functional;

// ---------------------------------------------------------------------------
// lref - given k ptr to module/model, return layer reference
// mref - given layer or k ptr, return reference to generic module
// mname_ - given module reference, return access to private, optional name
// mname  - given module reference return optional name
//        - also, given layer variant/layer ptr, return name or null ptr
// mlabel - demangle and simplify type name for use in error messages
// ---------------------------------------------------------------------------
Layer& lref(Ktag *g) {
 switch(g->a) {
  case Class::module: return ((Kmodule*)g)->m;
  case Class::model:  return ((Kmodel*)g)->m;
  default: AT_ERROR("unable to retrieve module from ",mapclass(g->a));
 }
}

Module& mref(const Layer& x) {return c10::visit(make_overload([](const auto& x)->Module& {return *x.ptr();}), x);}
Module& mref(Kmodule* x) {return mref(x->m);}     // module reference from Layer variant
Module& mref(Kmodel* x)  {return mref(x->m);}
Module& mref(Ktag *g)    {return mref(lref(g));}
Module& mref(Kloss* x)   {return *x->m.ptr();}    // module reference from AnyModule

const
c10::optional<std::string>& mname_(const Module& m) {return access_private::name_(m);}
c10::optional<std::string>& mname_(      Module& m) {return access_private::name_(m);}
S mname(const Module& m) {auto& s=access_private::name_(m); return const_cast<char*>(s ? (*s).c_str() : nullptr);}
S mname(const Layer& m) {return mname(mref(m));}
S mname(Kmodule* x) {return mname(x->m);}

std::string mlabel(const Module& x) {
 auto s=c10::demangle(typeid(x).name());
 if(!s.find("struct "))     s.erase(s.begin(),s.begin()+7);
 if(!s.find("class "))      s.erase(s.begin(),s.begin()+6);
 if(!s.find("torch::nn::")) s.erase(s.begin(),s.begin()+11);
 if(s.find("Impl",s.size()-4)==s.size()-4) s.erase(s.size()-4,s.size());
 return s;
}

// ----------------------------------------------------------------------------------
// OPTION - macro to append a module option to a k dictionary given dict,name & value
// argstart - return offset in k list to begin processing module args
// anymodule - forward declare function to create a module from k args, offset & cast
// mopt - forward declare function to return module settings as k dictionary
// ----------------------------------------------------------------------------------
#define OPTION(x,k,v) dictadd(x, mset(Setting::k), v)
static J argstart(K x,S s) {return xdict(x) ? -1 : (s ? 2 : 1);}
static AnyModule anymodule(K x,J i,Cast c);
static std::tuple<Cast,K> mopt(bool a,const Module& g);

// ----------------------------------------------------------------------------
// kmodule - allocate an object to store a pointer to a layer
// to - given layer & options, change device/data type
// ----------------------------------------------------------------------------
K kmodule(Cast c,const Layer& m) {return kptr(new Kmodule(c,m));}

K to(Kmodule* x,const TensorOptions& o,bool a) {
 auto s=torch::typeMetaToScalarType(o.dtype());
 auto& m=mref(x->m);
 if(o.has_device() && o.has_dtype()) m.to(o.device(),s,a);
 else if(o.has_device())             m.to(o.device(),a);
 else                                m.to(s,a);
 return (K)0;
}

// -----------------------------------------------------------------------------------
// msym - map to/from sym & enum for module, e.g. `conv3d <-> Cast::conv3d
// msyms - parse module and optional name symbol from k arg(s), throw error if not found
// mset - map to/from sym & enum for module options, e.g. `bias <-> Setting::bias
// mkeys - keys for dict/table of module state: `depth`module`name`options`parms`buffers
// -----------------------------------------------------------------------------------
static S msym(Cast c) {
 for(auto& m:env().module) if(c==std::get<1>(m)) return std::get<0>(m);
 AT_ERROR("unrecognized module: cannot translate enumeration ",(I)c," to symbol");
}

static Cast msym(S s) {
 for(auto& m:env().module) if(s==std::get<0>(m)) return std::get<1>(m);
 AT_ERROR("unrecognized module: ",s);
}

static void msyms(K x,S& s,S& nm) {
 nm=nullptr;
 if(x->t == -KS) {
  s=x->s;
 } else if(x->t == KS) {
  TORCH_CHECK(x->n>0, "module: empty symbol list");
  s=kS(x)[0];
  if(x->n>1) nm=kS(x)[1];
 } else if(!x->t) {
  TORCH_CHECK(x->n>0, "module: empty list");
  TORCH_CHECK(kK(x)[0]->t==-KS, "module: no symbol found, ",kstring(x));
  s=kK(x)[0]->s;
  if(x->n>1 && kK(x)[1]->t==-KS) nm=kK(x)[1]->s;
 } else {
  AT_ERROR("module: unrecognized arg(s), ", kstring(x));
 }
}

static S mset(Setting x) {
 for(auto& m:env().mset) if(x == std::get<1>(m)) return std::get<0>(m);
 AT_ERROR("unrecognized module option: ",(I)x);
}

static Setting mset(S x,Cast c=Cast::undefined);
static Setting mset(S x,Cast c) {
 for(auto& m:env().mset) if(x == std::get<0>(m)) return std::get<1>(m);
 if(c == Cast::undefined)
  AT_ERROR("unrecognized option: `",x);
 else
  AT_ERROR(msym(c),": unrecognized option `",x);
}

K mkeys(bool b) {
 K x=ktn(KS, b ? 6 : 4);
 kS(x)[0]=statekey(State::depth);
 kS(x)[1]=statekey(State::module);
 kS(x)[2]=statekey(State::name);
 kS(x)[3]=statekey(State::options);
 if(b) {
  kS(x)[4]=statekey(State::parms);
  kS(x)[5]=statekey(State::buffers);
 }
 return x;
}

// -----------------------------------------------------------------------------------
// seqlist - enlist x, only allow symbol scalar
// seq - convenience function to enlist all but 1st arg to build sequential arg list
// -----------------------------------------------------------------------------------
static K seqlist(K x) {
 K r;
 if(x->t<0) {
  TORCH_CHECK(x->t == -KS, "scalar expected to be a symbol, given a ",kname(x));
  r=ktn(KS,1), kS(r)[0]=x->s;
 } else {
  r=knk(1,r1(x));
 }
 return r;
}

KAPI seq(K x) {
 KTRY
  K r;
  if(x->t<0) {
   TORCH_CHECK(x->t==-KS, "seq: expecting module symbol, given ",kname(x),", ",kstring(x));
   r=r1(x);
  } else if(x->t>0) {
   TORCH_CHECK(x->t==KS, "seq: expecting module symbols, given ",kname(x),", ",kstring(x));
   TORCH_CHECK(x->n>0,   "seq: expecting at least one module symbol, given  empty list");
   r=ktn(0,x->n); kK(r)[0]=ks(kS(x)[0]);
   for(J i=1;i<x->n;++i) {
    kK(r)[i]=ktn(KS,1); kS(kK(r)[i])[0]=kS(x)[i];
   }
  } else {
   TORCH_CHECK(x->n>0, "seq: empty list");
   r=ktn(0,x->n);
   kK(r)[0]=r1(kK(x)[0]);
   for(J i=1;i<x->n;++i)
    kK(r)[i]=seqlist(kK(x)[i]);
  }
  return r;
 KCATCH("seq");
}

// -----------------------------------------------------------------------------------
// container - given module/module cast, return true if container module
// newcontainer - create new container (no options, parameters or buffers required)
// -----------------------------------------------------------------------------------
static bool container(Cast c) {
 switch(c) {
  case Cast::sequential:
  case Cast::seqnest:
  case Cast::seqjoin:
  case Cast::modulelist:
   return true;
  default: return false;
 }
}

static bool container(const Module& m) {
 if     (m.as<Sequential>()) return true;
 else if(m.as<SeqNest>())    return true;
 else if(m.as<SeqJoin>())    return true;
 else if(m.as<ModuleList>()) return true;
 else                        return false;
}

static bool container(const Layer& l) {return container(mref(l));}

static Layer newcontainer(Cast c) {
 switch(c) {
  case Cast::sequential:  return Sequential();
  case Cast::seqnest:     return SeqNest();
  case Cast::seqjoin:     return SeqJoin();
  case Cast::modulelist:  return ModuleList();
  default: AT_ERROR("unrecognized container: ", (I)c);
 }
}

// -------------------------------------------------------------------------------------------------
// rootlayer - given stack, pop back to root module, return as in struct to k
// makelayer - given generic module, convert back into Layer variant (for repopulating stack)
// mstack - given overall container layer, populate stack of all intermediate container layers
// -------------------------------------------------------------------------------------------------
static K rootlayer(Cast c,Layers& q) {
 while(q.size()>1) q.pop();
 return q.size() ? kmodule(c,q.top()) : (K)0;
}

static Layer makelayer(Module& m) {
 if     (m.as<Sequential>()) return Sequential(std::dynamic_pointer_cast<nn::SequentialImpl>(m.shared_from_this()));
 else if(m.as<SeqNest>())    return    SeqNest(std::dynamic_pointer_cast<SeqNestImpl>(m.shared_from_this()));
 else if(m.as<SeqJoin>())    return    SeqJoin(std::dynamic_pointer_cast<SeqJoinImpl>(m.shared_from_this()));
 else if(m.as<ModuleList>()) return ModuleList(std::dynamic_pointer_cast<nn::ModuleListImpl>(m.shared_from_this()));
 else AT_ERROR("unable to create parent layer from ",m.name());
}

static void mstack(size_t d,Module& m,Layers& q) {
 while(q.size()>d) q.pop();
 if(container(m)) {
  q.push(makelayer(m));
  for(auto& i:m.children())
   mstack(d+1,*i,q);
 }
}

static void mstack(size_t d,Layer& l,Layers& q) {
 while(q.size()>d) q.pop();
 auto& m=mref(l);
 if(container(m)) {
  q.push(l);
  for(auto& i:m.children())
   mstack(d+1,*i,q);
 }
}

//static void mstack(Kmodule *l,Layers& q) {mstack(0,mref(l->m),q);} // initialize stack
static void mstack(Kmodule *l,Layers& q) {mstack(0,l->m,q);} // initialize stack

// ------------------------------------------------------------------------------------
// mforward - given layer, run forward calc on tensor x and optional y,z tensors
// ------------------------------------------------------------------------------------
Tensor mforward(Layer& m,const Tensor& x,const Tensor& y,const Tensor& z) {
 return c10::visit(
  make_overload(
   [&x,&y,&z](AnyModule& m) {
    return y.defined() ? (z.defined() ? m.forward(x,y,z) : m.forward(x,y)) : m.forward(x);
   },
   [&x,&y,&z](SeqJoin& m) {
    TORCH_CHECK(x.defined() && y.defined(), "seqjoin: forward calculation expects two tensors, x & y");
    TORCH_CHECK(!z.defined(), "seqjoin: unexpected 3rd tensor supplied to forward calculation");
    return m->forward(x,y);
   },
   [](ModuleList& m) { AT_ERROR("No forward function for ModuleList"); return Tensor();},
   [&x,&y,&z](auto& m) {
    return y.defined() ? (z.defined() ? m->forward(x,y,z) : m->forward(x,y)) : m->forward(x);
   }
  ), m);
}

K mforward(Layer& m,K a) {
 Tensor x,y,z; TensorVector *v;
 if((v=xvec(a,1))) {
  TORCH_CHECK(v->size(), "forward: empty vector of tensors supplied");
  IntArrayRef i;
  if(a->n==2) {
   return kten(mforward(m, v->at(0)));
  } else if(a->n==3 && xsize(a,2,i)) {
   switch(i.size()) {
    case 1: return kten(mforward(m, v->at(i[0])));
    case 2: return kten(mforward(m, v->at(i[0]), v->at(i[1])));
    case 3: return kten(mforward(m, v->at(i[0]), v->at(i[1]), v->at(i[2])));
    default: AT_ERROR("forward: vector w'indices expects 1-3 indices, ",i.size()," supplied");
   }
  } else {
   AT_ERROR("forward with vector expects format of (module/model;vector) or (module/model;vector;indices)");
  }
 } else {
  TORCH_CHECK(!a->t && a->n>1 && a->n<5, "forward expects 2-4 args: module & model and up to 3 tensors/arrays, e.g. (m;x) or (m;x;y;z)");
  if(!xten(a,1,x))            x=kput(a,1);
  if(a->n>=3 && !xten(a,2,y)) y=kput(a,2);
  if(a->n==4 && !xten(a,3,z)) z=kput(a,3);
  return kten(mforward(m,x,y,z));
 }
}

// ---------------------------------------------------------------------------------------
// mattr - return requested attribute of given layer
// ---------------------------------------------------------------------------------------
K mattr(const Layer& l,Ktype k,Attr a) {
 switch(a) {
  case Attr::ptr: return kj((intptr_t)mref(l).shared_from_this().get());
  case Attr::ref: return kj(mref(l).shared_from_this().use_count()-1);
  default: AT_ERROR(mapattr(a),": not implemented for module");
 }
}

// ----------------------------------------------------------------------------------------------------
// covers of input checking fns with error msg specific to module settings and module names:
// ----------------------------------------------------------------------------------------------------
// mbool - check positional args or name-value pairs for boolean, else error w'module & option name
// mode  - check positional args of name-value pairs for symbol,  else error w'module & option name
// int64 - check positional args or name-value pairs for long int, else error w'module & option
// int64n - int64 but returns optional, i.e. nullopt if k value is null
// mdouble - check for double(or long) from positional or name-value pair arg
// optdouble - call mdouble() but return null if k null supplied
// ----------------------------------------------------------------------------------------------------
static bool mbool(K x,J i,Cast c,Setting s) {
 bool b;
 TORCH_CHECK(xbool(x,i,b), msym(c)," ",mset(s),": expected boolean scalar, given ",kname(x,i));
 return b;
}

static bool mbool(const Pairs& p,Cast c) {
 TORCH_CHECK(p.t==-KB, msym(c)," ",p.k,": expected boolean scalar, given ",kname(p.t));
 return p.b;
}

static S code(K x,J i,Cast c,Setting s) {
 S m;
 TORCH_CHECK(xsym(x,i,m), msym(c)," ",mset(s),": expected symbol, given ",kname(x,i));
 return m;
}

static S code(const Pairs& p,Cast c) {
 TORCH_CHECK(p.t==-KS, msym(c)," ",p.k,": expected symbol, given ",kname(p.t));
 return p.s;
}

static int64_t int64(K x,J i,Cast c,Setting s) {
 int64_t n;
 TORCH_CHECK(xint64(x,i,n), msym(c)," ",mset(s),": expected long scalar, given ",kname(x,i));
 return n;
}

static int64_t int64(const Pairs& p,Cast c) {
 TORCH_CHECK(p.t==-KJ, msym(c)," ",p.k,": expected long scalar, given ",kname(p.t));
 return p.j;
}

static c10::optional<int64_t> int64n(K x,J i,Cast c,Setting s) {auto n=int64(x,i,c,s); if(null(n)) return c10::nullopt; else return n;}
static c10::optional<int64_t> int64n(const Pairs& p,Cast c)    {auto n=int64(p,c);     if(null(n)) return c10::nullopt; else return n;}

static double mdouble(K x,J i,Cast c,Setting s) {
 double f;
 TORCH_CHECK(xnum(x,i,f), msym(c)," ",mset(s),": expected double, given ",kname(x,i));
 return f;
}

static double mdouble(const Pairs& p,Cast c) {
 TORCH_CHECK(p.t==-KJ || p.t==-KF, msym(c)," ",p.k,": expected double, given ",kname(p.t));
 return pdouble(p);
}

static c10::optional<double> optdouble(K x,J i,Cast c,Setting s) {double d=mdouble(x,i,c,s); if(d==d) return d; else return c10::nullopt;}
static c10::optional<double> optdouble(const Pairs& p,Cast c)    {double d=mdouble(p,c);     if(d==d) return d; else return c10::nullopt;}

// ----------------------------------------------------------------------------------------------------
// mlongs - check for long(s), return vector else error specific to module and setting
// mdoubles - check for double(s), return vector else error specific to module and setting
// ----------------------------------------------------------------------------------------------------
static LongVector mlongs(K x,J i,Cast c,Setting s) {
 IntArrayRef a;
 TORCH_CHECK(xsize(x,i,a), msym(c)," ",mset(s),": expected long(s), given ",kname(x,i));
 return a.vec();
}

static LongVector mlongs(const Pairs& p,Cast c) {
 IntArrayRef a;
 TORCH_CHECK(p.t==-KJ || p.t==KJ, msym(c)," ",p.k,": expected long(s), given ",kname(p.t));
 psize(p,a);
 return a.vec();
}

static DoubleVector mdoubles(K x,J i,Cast c,Setting s) {
 J n; F *f; IntArrayRef a; DoubleVector v;
 if(xsize(x,i,a)) {
  for(auto j:a) v.push_back(j);
 } else if(xdouble(x,i,n,f)) {
  v=DoubleArrayRef(f,n).vec();
 } else {
  AT_ERROR(msym(c)," ",mset(s),": expected double(s), given ",kname(x,i));
 }
 return v;
}

static DoubleVector mdoubles(const Pairs& p,Cast c) {
 DoubleVector v;
 if(p.t==-KJ || p.t==KJ) {
  IntArrayRef a; psize(p,a);
  for(auto j:a) v.push_back(j);
 } else if(p.t==-KF || p.t==KF) {
  DoubleArrayRef a; pdoubles(p,a); v=a.vec();
 } else {
  AT_ERROR(msym(c)," ",p.k,": expected double(s), given ",kname(p.t));
 }
 return v;
}


// -------------------------------------------------------------------------------------------------
// exarray - check positional or name-value args for long(s), return expanding array,  else error
// exoptional - similar to exarray, for optional long(s), return expanding array with nulls
// exdouble - similar to exarray, but for double array
// -------------------------------------------------------------------------------------------------
template<size_t D> static ExpandingArray<D> exarray(K a,J i,Cast c,Setting s) {
 K x=kK(a)[i];
 TORCH_CHECK(x->t==-KJ || x->t==KJ, msym(c)," ",mset(s),": expected long(s), given ",kname(x->t));
 TORCH_CHECK(x->t==-KJ || x->n==D,  msym(c)," ",mset(s),": expected scalar or ",D,"-element input, given ",x->n,"-element list");
 if(x->t==-KJ)
  return ExpandingArray<D>(x->j);
 else
  return ExpandingArray<D>(IntArrayRef((int64_t*)kJ(x),x->n));
}

template<size_t D> static ExpandingArray<D> exarray(const Pairs& p,Cast c) {
 TORCH_CHECK(p.t==-KJ || p.t==KJ,   msym(c)," ",p.k,": expected long(s), given ",kname(p.t));
 TORCH_CHECK(p.t==-KJ || p.v->n==D, msym(c)," ",p.k,": expected scalar or ",D,"-element input, given ",p.v->n,"-element list");
 if(p.t==-KJ)
  return ExpandingArray<D>(p.j);
 else
  return ExpandingArray<D>(IntArrayRef((int64_t*)kJ(p.v),p.v->n));
}

template<size_t D> static Exoptional<D> exoptional(J j) {
 return null(j) ? Exoptional<D>(c10::nullopt) : Exoptional<D>(j);
}

template<size_t D> static Exoptional<D> exoptional(K x) {
 auto a=Exoptional<D>(IntArrayRef((int64_t*)kJ(x),x->n));
 for(J i=0;i<x->n;++i) if(null((*a)[i].value())) (*a)[i]=c10::nullopt;
 return a;
}

template<size_t D> static Exoptional<D> exoptional(K a,J i,Cast c,Setting s) {
 K x=kK(a)[i];
 TORCH_CHECK(x->t==-KJ || x->t==KJ, msym(c)," ",mset(s),": expected long(s), given ",kname(x->t));
 TORCH_CHECK(x->t==-KJ || x->n==D,  msym(c)," ",mset(s),": expected scalar or ",D,"-element input, given ",x->n,"-element list");
 return x->t == -KJ ? exoptional<D>(x->j) : exoptional<D>(x);
}

template<size_t D> static Exoptional<D> exoptional(const Pairs& p,Cast c) {
 TORCH_CHECK(p.t==-KJ || p.t==KJ,   msym(c)," ",p.k,": expected long(s), given ",kname(p.t));
 TORCH_CHECK(p.t==-KJ || p.v->n==D, msym(c)," ",p.k,": expected scalar or ",D,"-element input, given ",p.v->n,"-element list");
 return p.t == -KJ ? exoptional<D>(p.j) : exoptional<D>(p.v);
}

template<size_t D> static Exdouble<D> exdouble(K a,J i,Cast c,Setting s) {
 K x=kK(a)[i];
 TORCH_CHECK(x->t==-KF || x->t==KF, msym(c)," ",mset(s),": expected double(s), given ",kname(x->t));
 TORCH_CHECK(x->t==-KF || x->n==D,  msym(c)," ",mset(s),": expected scalar or ",D,"-element input, given ",x->n,"-element list");
 if(x->t==-KF)
  return Exdouble<D>(x->f);
 else
  return Exdouble<D>(torch::ArrayRef<double>(kF(x),x->n));
}

template<size_t D> static Exdouble<D> exdouble(const Pairs& p,Cast c) {
 TORCH_CHECK(p.t==-KF || p.t==KF,   msym(c)," ",p.k,": expected double(s), given ",kname(p.t));
 TORCH_CHECK(p.t==-KF || p.v->n==D, msym(c)," ",p.k,": expected scalar or ",D,"-element input, given ",p.v->n,"-element list");
 if(p.t==-KF)
  return Exdouble<D>(p.f);
 else
  return Exdouble<D>(torch::ArrayRef<double>(kF(p.v),p.v->n));
}

// --------------------------------------------------------------------------------------
// batchnorm - get/set batch norm & instance norm options
//             both module types and dimensions(1,2,3d) use the same options structure
//             except batch norm's momentum is an optional double
// --------------------------------------------------------------------------------------
static double momentum(c10::optional<double> x) {return x.value();}

template<typename O> static O batchnorm(K x,J i,Cast c) {
 O o(0);
 bool in=false; Pairs p; J n=xargc(x,i,p);
 for(J j=0;j<n;++j)
   switch(j) {
    case 0: o.num_features(int64(x,i+j,c,Setting::in)); in=true; break;
    case 1: o.eps(mdouble(x,i+j,c,Setting::eps));break;
    case 2: o.momentum(mdouble(x,i+j,c,Setting::momentum)); break;
    case 3: o.affine(mbool(x,i+j,c,Setting::affine)); break;
    case 4: o.track_running_stats(mbool(x,i+j,c,Setting::affine)); break;
    default: AT_ERROR(msym(c),": up to 5 positional arguments expected, ",n," given");
  }
 while(xpair(p))
  switch(mset(p.k,c)) {
   case Setting::in:       o.num_features(int64(p,c)); in=true; break;
   case Setting::eps:      o.eps(mdouble(p,c)); break;
   case Setting::momentum: o.momentum(mdouble(p,c)); break;
   case Setting::affine:   o.affine(mbool(p,c)); break;
   case Setting::track:    o.track_running_stats(mbool(p,c)); break;
   default: AT_ERROR("unrecognized ",msym(c)," option: ",p.k); break;
  }
 TORCH_CHECK(in,msym(c),": number of input features not defined");
 return o;
}

template<typename O> static K batchnorm(bool a,const O& o) {
 K x=KDICT; O d(o.num_features());
 OPTION(x, in, kj(o.num_features()));
 if(a || (o.eps()      != d.eps()))      OPTION(x, eps,       kf(o.eps()));
 if(a || (o.momentum() != d.momentum())) OPTION(x, momentum,  kf(momentum(o.momentum())));
 if(a || (o.affine()   != d.affine()))   OPTION(x, affine,    kb(o.affine()));
 if(a || (o.track_running_stats() != d.track_running_stats())) OPTION(x, track, kb(o.track_running_stats()));
 return x;
}

// -------------------------------------------------------------------------------------
// localnorm - local response norm, cross map 2d norm, get/set options size,alpha,beta,k
// -------------------------------------------------------------------------------------
template<typename O> static O localnorm(K x,J i,Cast c) {
 O o(0);
 bool b=c==Cast::localnorm,sz=false; Pairs p; J n=xargc(x,i,p);
 for(J j=0;j<n;++j)
   switch(j) {
    case 0: o.size(int64(x,i+j,c,Setting::size)); sz=true; break;
    case 1: o.alpha(mdouble(x,i+j,c,Setting::alpha)); break;
    case 2: o.beta(mdouble(x,i+j,c,Setting::beta)); break;
    case 3: b ? o.k(mdouble(x,i+j,c,Setting::k)) : o.k(int64(x,i+j,c,Setting::k)); break;
    default: AT_ERROR(msym(c),": up to 4 positional arguments expected, ",n," given");
  }
 while(xpair(p))
  switch(mset(p.k,c)) {
   case Setting::size:  o.size(int64(p,c)); sz=true; break;
   case Setting::alpha: o.alpha(mdouble(p,c)); break;
   case Setting::beta:  o.beta(mdouble(p,c)); break;
   case Setting::k:     b ? o.k(mdouble(p,c)) : o.k(int64(p,c)); break;
   default: AT_ERROR("unrecognized ",msym(c)," option: ",p.k); break;
  }
 TORCH_CHECK(sz, msym(c),": specify no. of neighboring channels to use for normalization");
 return o;
}

template<typename O> static K localnorm(bool a,Cast c,const O& o) {
 K x=KDICT; O d(o.size());
 OPTION(x, size, kj(o.size()));
 if(a || (o.alpha() != d.alpha())) OPTION(x, alpha, kf(o.alpha()));
 if(a || (o.beta()  != d.beta()))  OPTION(x, beta,  kf(o.beta()));
 if(a || (o.k()     != d.k()))     OPTION(x, k,     c==Cast::localnorm ? kf(o.k()) : kj(o.k()));
 return x;
}

// --------------------------------------------------------------------------------------
// groupnorm - group norm, get/set number of groups,channels,eps,affine flag
// --------------------------------------------------------------------------------------
static nn::GroupNormOptions groupnorm(K x,J i,Cast c) {
 nn::GroupNormOptions o(0,0);
 bool g=false,h=false; Pairs p; J n=xargc(x,i,p);
 for(J j=0;j<n;++j)
   switch(j) {
    case 0: o.num_groups(int64(x,i+j,c,Setting::groups)); g=true; break;
    case 1: o.num_channels(int64(x,i+j,c,Setting::channels)); h=true; break;
    case 2: o.eps(mdouble(x,i+j,c,Setting::eps)); break;
    case 3: o.affine(mbool(x,i+j,c,Setting::affine)); break;
    default: AT_ERROR(msym(c),": up to 4 positional arguments expected, ",n," given");
  }
 while(xpair(p))
  switch(mset(p.k,c)) {
   case Setting::groups:   o.num_groups(int64(p,c)); g=true; break;
   case Setting::channels: o.num_channels(int64(p,c)); h=true; break;
   case Setting::eps:      o.eps(mdouble(p,c)); break;
   case Setting::affine:   o.affine(mbool(p,c)); break;
   default: AT_ERROR("unrecognized ",msym(c)," option: ",p.k); break;
  }
 TORCH_CHECK(g, msym(c),": specify no. of groups to separate the channels into");
 TORCH_CHECK(h, msym(c),": specify no. of channels expected in input");
 return o;
}

static K groupnorm(bool a,const nn::GroupNormOptions& o) {
 K x=KDICT; nn::GroupNormOptions d(o.num_groups(),o.num_channels());
 OPTION(x, groups,   kj(o.num_groups()));
 OPTION(x, channels, kj(o.num_channels()));
 if(a || (o.eps()    != d.eps()))    OPTION(x, eps,    kf(o.eps()));
 if(a || (o.affine() != d.affine())) OPTION(x, affine, kb(o.affine()));
 return x;
}

// --------------------------------------------------------------------------------------
// layernorm - get/set shape,eps,affine flag for layer normalization
// --------------------------------------------------------------------------------------
static nn::LayerNormOptions layernorm(K x,J i,Cast c) {
 nn::LayerNormOptions o({}); Pairs p; J n=xargc(x,i,p);
 for(J j=0;j<n;++j)
   switch(j) {
    case 0: o.normalized_shape(mlongs(x,i+j,c,Setting::shape)); break;
    case 1: o.eps(mdouble(x,i+j,c,Setting::eps)); break;
    case 2: o.elementwise_affine(mbool(x,i+j,c,Setting::affine)); break;
    default: AT_ERROR(msym(c),": up to 3 positional arguments expected, ",n," given");
  }
 while(xpair(p))
  switch(mset(p.k,c)) {
   case Setting::shape:  o.normalized_shape(mlongs(p,c)); break;
   case Setting::eps:    o.eps(mdouble(p,c)); break;
   case Setting::affine: o.elementwise_affine(mbool(p,c)); break;
   default: AT_ERROR("unrecognized ",msym(c)," option: ",p.k); break;
  }
 TORCH_CHECK(o.normalized_shape().size(), msym(c),": no normalized shape given");
 return o;
}

static K layernorm(bool a,const nn::LayerNormOptions& o) {
 K x=KDICT; nn::LayerNormOptions d(o.normalized_shape());
 OPTION(x, shape, klist(o.normalized_shape().size(),o.normalized_shape().data()));
 if(a || (o.eps()    != d.eps())) OPTION(x, eps, kf(o.eps()));
 if(a || (o.elementwise_affine() != d.elementwise_affine())) OPTION(x, affine, kb(o.elementwise_affine()));
 return x;
}

// --------------------------------------------------------------------------------------
// normalize - pytorch has functional form only
// --------------------------------------------------------------------------------------
static fnn::NormalizeFuncOptions normalize(K x,J i,Cast c,Tensor& r) {
 Pairs p; J n=xargc(x,i,p); fnn::NormalizeFuncOptions o;
 if(n>0 && xten(x,i+n-1,r)) n--;
 for(J j=0;j<n;++j)
  switch(j) {
   case 0: o.p(mdouble(x,i+j,c,Setting::p)); break;
   case 1: o.dim(int64(x,i+j,c,Setting::dim)); break;
   case 2: o.eps(mdouble(x,i+j,c,Setting::eps)); break;
   default: AT_ERROR(msym(c),": unrecognized positional arg(s), up to 4 args(p,dim,eps,output tensor) expected, ",n," supplied");
  }
 while(xpair(p))
  switch(mset(p.k,c)) {
   case Setting::p: o.p(mdouble(p,c)); break;
   case Setting::dim: o.dim(int64(p,c)); break;
   case Setting::eps: o.eps(mdouble(p,c)); break;
   case Setting::out: if(!pempty(p)) pten(p,r);
   default: AT_ERROR("unrecognized option: ",p.k," for normalize");
  }
 if(r.defined()) 
  o.out(r);
 return o;
}

KAPI Normalize(K x) {
 KTRY
  namespace f=fnn;
  Tensor r,*t=nullptr;
  if(x->t || (t=xten(x))) {
   return kresult(t, f::normalize(t ? *t : kput(x), f::NormalizeFuncOptions()));
  } else {
   t=xten(x,0);
   return kresult(t||r.defined(), f::normalize(t ? *t : kput(x,0), normalize(x,1,Cast::normalize,r)));
  }
 KCATCH("normalize");
}

// --------------------------------------------------------------------------------------
// convpad - translate symbol to variant used for padding mode
// conv - create 1-3d convolution, set dictionary given module
//        with version 1.4, the c++ ConvImpl class was split into regular & transposed
//        ConvOptions & ConvTransOptions have different members, 
// convtran - similar to conv() except adds output_padding and changes position order
// --------------------------------------------------------------------------------------
static nn::detail::conv_padding_mode_t convpad(S s) {
 switch(emap(s)) {
  case Enum::zeros:     return torch::kZeros;
  case Enum::reflect:   return torch::kReflect;
  case Enum::replicate: return torch::kReplicate;
  case Enum::circular:  return torch::kCircular;
  default: AT_ERROR("unrecognized padding mode: ",s); break;
 }
}

template<size_t D> static nn::ConvOptions<D> conv(K x,J i,Cast c) {
 nn::ConvOptions<D> o(0,0,0);
 bool in=false,out=false,sz=false; Pairs p; J n=xargc(x,i,p);
 for(J j=0;j<n;++j)
   switch(j) {
    case 0: o.in_channels (int64(x,i+j,c,Setting::in));        in=true; break;
    case 1: o.out_channels(int64(x,i+j,c,Setting::in));       out=true; break;
    case 2: o.kernel_size (exarray<D>(x,i+j,c,Setting::size)); sz=true; break;
    case 3: o.stride      (exarray<D>(x,i+j,c,Setting::stride));   break;
    case 4: o.padding     (exarray<D>(x,i+j,c,Setting::pad));      break;
    case 5: o.dilation    (exarray<D>(x,i+j,c,Setting::dilate));   break;
    case 6: o.groups      (int64(x,i+j,c,Setting::groups));        break;
    case 7: o.bias        (mbool    (x,i+j,c,Setting::bias));      break;
    case 8: o.padding_mode(convpad(code(x,i+j,c,Setting::padmode))); break;
    default: AT_ERROR(msym(c),": up to 9 positional arguments expected, ",n," given");
  }
 while(xpair(p))
  switch(mset(p.k,c)) {
   case Setting::in:        o.in_channels (int64(p,c));     in=true; break;
   case Setting::out:       o.out_channels(int64(p,c));    out=true; break;
   case Setting::size:      o.kernel_size (exarray<D>(p,c)); sz=true; break;
   case Setting::stride:    o.stride      (exarray<D>(p,c)); break;
   case Setting::pad:       o.padding     (exarray<D>(p,c)); break;
   case Setting::dilate:    o.dilation    (exarray<D>(p,c)); break;
   case Setting::groups:    o.groups      (int64(p,c));     break;
   case Setting::bias:      o.bias        (mbool(p,c));     break;
   case Setting::padmode:   o.padding_mode(convpad(code(p,c)));   break;
   default: AT_ERROR("unrecognized convolution option: ",p.k); break;
  }
 TORCH_CHECK(in,  msym(c),": number of input channels not defined");
 TORCH_CHECK(out, msym(c),": number of output channels not defined");
 TORCH_CHECK(sz,  msym(c),": no kernel size(s) given");
 return o;
}

template<size_t D> static nn::ConvTransposeOptions<D> convtran(K x,J i,Cast c) {
 nn::ConvTransposeOptions<D> o(0,0,0);
 bool in=false,out=false,sz=false; Pairs p; J n=xargc(x,i,p);
 for(J j=0;j<n;++j)
   switch(j) {
    case 0: o.in_channels   (int64(x,i+j,c,Setting::in));        in=true; break;
    case 1: o.out_channels  (int64(x,i+j,c,Setting::in));       out=true; break;
    case 2: o.kernel_size   (exarray<D>(x,i+j,c,Setting::size)); sz=true; break;
    case 3: o.stride        (exarray<D>(x,i+j,c,Setting::stride)); break;
    case 4: o.padding       (exarray<D>(x,i+j,c,Setting::pad));    break;
    case 5: o.output_padding(exarray<D>(x,i+j,c,Setting::outpad)); break;
    case 6: o.groups        (int64(x,i+j,c,Setting::groups));      break;
    case 7: o.bias          (mbool(x,i+j,c,Setting::bias));        break;
    case 8: o.dilation      (exarray<D>(x,i+j,c,Setting::dilate)); break;
    case 9: o.padding_mode  (convpad(code(x,i+j,c,Setting::padmode))); break;
    default: AT_ERROR(msym(c),": up to 9 positional arguments expected, ",n," given");
  }
 while(xpair(p))
  switch(mset(p.k,c)) {
   case Setting::in:        o.in_channels   (int64(p,c));      in=true; break;
   case Setting::out:       o.out_channels  (int64(p,c));     out=true; break;
   case Setting::size:      o.kernel_size   (exarray<D>(p,c)); sz=true; break;
   case Setting::stride:    o.stride        (exarray<D>(p,c)); break;
   case Setting::pad:       o.padding       (exarray<D>(p,c)); break;
   case Setting::outpad:    o.output_padding(exarray<D>(p,c)); break;
   case Setting::groups:    o.groups        (int64(p,c));      break;
   case Setting::bias:      o.bias          (mbool(p,c));      break;
   case Setting::dilate:    o.dilation      (exarray<D>(p,c)); break;
   case Setting::padmode:   o.padding_mode(convpad(code(p,c)));break;
   default: AT_ERROR("unrecognized convolution option: ",p.k); break;
  }
 TORCH_CHECK(in,  msym(c), ": number of input channels not defined");
 TORCH_CHECK(out, msym(c), ": number of output channels not defined");
 TORCH_CHECK(sz,  msym(c), ": no kernel size(s) given");
 return o;
}

template<size_t D> static K conv(bool a,const nn::detail::ConvNdOptions<D>& o) {
 K x=KDICT; nn::detail::ConvNdOptions<D> d(o.in_channels(),o.out_channels(),o.kernel_size());
 bool t=o.transposed();
 OPTION(x, in,   kj(o.in_channels()));
 OPTION(x, out,  kj(o.out_channels()));
 OPTION(x, size, KEX(o.kernel_size()));
 if(a || (*o.stride()  != *d.stride()))  OPTION(x, stride, KEX(o.stride()));
 if(a || (*o.padding() != *d.padding())) OPTION(x, pad,    KEX(o.padding()));
 if(t) {
  if(a || (*o.output_padding() != *d.output_padding())) OPTION(x, outpad, KEX(o.output_padding()));
 } else {
  if(a || (*o.dilation() != *d.dilation())) OPTION(x, dilate, KEX(o.dilation()));
 }
 if(a || ( o.groups()    !=  d.groups())) OPTION(x, groups, kj(o.groups()));
 if(a || ( o.bias()      !=  d.bias()))   OPTION(x, bias,   kb(o.bias()));
 if(t) {
  if(a || (*o.dilation() != *d.dilation())) OPTION(x, dilate, KEX(o.dilation()));
 }
 if(a || o.padding_mode().index() != d.padding_mode().index()) OPTION(x, padmode, ks(ESYM(o.padding_mode())));
 return x;
}

// --------------------------------------------------------------------------------------
// fold,unfold - set/get size,dilation,padding,stride
// --------------------------------------------------------------------------------------
static nn::FoldOptions fold(K x,J i,Cast c) {
 nn::FoldOptions o(0,0);
 bool out=false,sz=false; Pairs p; J n=xargc(x,i,p);
 for(J j=0;j<n;++j)
   switch(j) {
    case 0: o.output_size(exarray<2>(x,i+j,c,Setting::out)); out=true; break;
    case 1: o.kernel_size(exarray<2>(x,i+j,c,Setting::size)); sz=true; break;
    case 2: o.dilation   (exarray<2>(x,i+j,c,Setting::dilate)); break;
    case 3: o.padding    (exarray<2>(x,i+j,c,Setting::pad));    break;
    case 4: o.stride     (exarray<2>(x,i+j,c,Setting::stride)); break;
    default: AT_ERROR(msym(c),": up to 5 positional arguments expected, ",n," given");
  }
 while(xpair(p))
  switch(mset(p.k,c)) {
   case Setting::out:       o.output_size(exarray<2>(p,c));out=true; break;
   case Setting::size:      o.kernel_size(exarray<2>(p,c)); sz=true; break;
   case Setting::dilate:    o.dilation   (exarray<2>(p,c)); break;
   case Setting::pad:       o.padding    (exarray<2>(p,c)); break;
   case Setting::stride:    o.stride     (exarray<2>(p,c)); break;
   default: AT_ERROR("unrecognized fold option: ",p.k); break;
  }
 TORCH_CHECK(out, msym(c),": no output size given");
 TORCH_CHECK(sz,  msym(c),": no kernel size given");
 return o;
}

static K fold(bool a,const nn::FoldOptions& o) {
 K x=KDICT; nn::FoldOptions d(o.output_size(),o.kernel_size());
 OPTION(x, out,  KEX(o.output_size()));
 OPTION(x, size, KEX(o.kernel_size()));
 if(a || (*o.dilation() != *d.dilation())) OPTION(x, dilate, KEX(o.dilation()));
 if(a || (*o.padding()  != *d.padding()))  OPTION(x, pad,    KEX(o.padding()));
 if(a || (*o.stride()   != *d.stride()))   OPTION(x, stride, KEX(o.stride()));
 return x;
}

static nn::UnfoldOptions unfold(K x,J i,Cast c) {
 nn::UnfoldOptions o(0);
 bool sz=false; Pairs p; J n=xargc(x,i,p);
 for(J j=0;j<n;++j)
   switch(j) {
    case 0: o.kernel_size(exarray<2>(x,i+j,c,Setting::size)); sz=true; break;
    case 1: o.dilation   (exarray<2>(x,i+j,c,Setting::dilate)); break;
    case 2: o.padding    (exarray<2>(x,i+j,c,Setting::pad));    break;
    case 3: o.stride     (exarray<2>(x,i+j,c,Setting::stride)); break;
    default: AT_ERROR(msym(c),": up to 4 positional arguments expected, ",n," given");
  }
 while(xpair(p))
  switch(mset(p.k,c)) {
   case Setting::size:      o.kernel_size(exarray<2>(p,c)); sz=true; break;
   case Setting::dilate:    o.dilation   (exarray<2>(p,c)); break;
   case Setting::pad:       o.padding    (exarray<2>(p,c)); break;
   case Setting::stride:    o.stride     (exarray<2>(p,c)); break;
   default: AT_ERROR("unrecognized unfold option: ",p.k); break;
  }
 TORCH_CHECK(sz, msym(c),": no kernel size given");
 return o;
}

static K unfold(bool a,const nn::UnfoldOptions& o) {
 K x=KDICT; nn::UnfoldOptions d(o.kernel_size());
 OPTION(x, size, KEX(o.kernel_size()));
 if(a || (*o.dilation() != *d.dilation())) OPTION(x, dilate, KEX(o.dilation()));
 if(a || (*o.padding()  != *d.padding()))  OPTION(x, pad,    KEX(o.padding()));
 if(a || (*o.stride()   != *d.stride()))   OPTION(x, stride, KEX(o.stride()));
 return x;
}

static K kfold(K x,Cast c) {
 KTRY
  TORCH_CHECK(!x->t, msym(c)," not implemented for ",kname(x->t));
  Tensor r, *t=xten(x,0);
  return kresult(t, c==Cast::fold
       ? fnn::fold  (t ? *t : kput(x,0),   fold(x,1,c))
       : fnn::unfold(t ? *t : kput(x,0), unfold(x,1,c)));
 KCATCH("fold");
}

KAPI   Fold(K x) {return kfold(x, Cast::fold);}
KAPI Unfold(K x) {return kfold(x, Cast::unfold);}

// --------------------------------------------------------------------------------------
// upsample & interpolate - both require same options, interpolate has additional flag
// upsample is implemented as a module in pytorch, interpolate as a function
// --------------------------------------------------------------------------------------
static void upmode(nn::UpsampleOptions& o,S s) {
 switch(emap(s)) {
  case Enum::nearest:   o.mode(torch::kNearest); break;
  case Enum::linear:    o.mode(torch::kLinear); break;
  case Enum::bilinear:  o.mode(torch::kBilinear); break;
  case Enum::bicubic:   o.mode(torch::kBicubic); break;
  case Enum::trilinear: o.mode(torch::kTrilinear); break;
  default: AT_ERROR("unrecognized upsample mode: ",s); break;
 }
}

static void upmode(fnn::InterpolateFuncOptions& o,S s) {
 switch(emap(s)) {
  case Enum::nearest:   o.mode(torch::kNearest); break;
  case Enum::linear:    o.mode(torch::kLinear); break;
  case Enum::bilinear:  o.mode(torch::kBilinear); break;
  case Enum::bicubic:   o.mode(torch::kBicubic); break;
  case Enum::trilinear: o.mode(torch::kTrilinear); break;
  case Enum::area:      o.mode(torch::kArea); break;
  default: AT_ERROR("unrecognized interpolate mode: ",s); break;
 }
}

// recompute_scale_factor only part of interpolate options, separate fns to handle setting
static void rescale(K x,J i,Cast c,Setting s,nn::UpsampleOptions& o) {
 AT_ERROR(msym(c),": up to 4 positional arguments expected, 5th argument unrecognized");
}
static void rescale(K x,J i,Cast c,Setting s,fnn::InterpolateFuncOptions& o) {
 if(xempty(x,i)) o.recompute_scale_factor({}); else o.recompute_scale_factor(mbool(x,i,c,s));
}
static void rescale(Pairs& p,Cast c,nn::UpsampleOptions& o) {
 AT_ERROR(msym(c),": rescale is not a recognized option");
}
static void rescale(Pairs& p,Cast c,fnn::InterpolateFuncOptions& o) {
 if(pempty(p)) o.recompute_scale_factor({}); else o.recompute_scale_factor(mbool(p,c));
}

template<typename O>static O upsample(K x,J i,Cast c) {
 O o; Pairs p; J n=xargc(x,i,p);
 for(J j=0;j<n;++j) {
  switch(j) {
   case 0: if(xempty(x,i+j)) o.size({}); else o.size(mlongs(x,i+j,c,Setting::size)); break;
   case 1: if(xempty(x,i+j)) o.scale_factor({}); else o.scale_factor(mdoubles(x,i+j,c,Setting::scale)); break;
   case 2: upmode(o,code(x,i+j,c,Setting::mode)); break;
   case 3: if(xempty(x,i+j)) o.align_corners({}); else o.align_corners(mbool(x,i+j,c,Setting::align)); break;
   case 4: rescale(x,i+j,c,Setting::rescale,o); break;
   default: AT_ERROR(msym(c),": up to ",(c==Cast::upsample ? 4 : 5)," positional arguments expected, ",n," given");
  }
 }
 while(xpair(p))
  switch(mset(p.k,c)) {
   case Setting::size:    if(pempty(p)) o.size({}); else o.size(mlongs(p,c)); break;
   case Setting::scale:   if(pempty(p)) o.scale_factor({}); else o.scale_factor(mdoubles(p,c)); break;
   case Setting::mode:    upmode(o,code(p,c)); break;
   case Setting::align:   if(pempty(p)) o.align_corners({}); else o.align_corners(mbool(p,c)); break;
   case Setting::rescale: rescale(p,c,o); break;
   default: AT_ERROR("unrecognized ",msym(c)," option: ",p.k); break;
  }
 if(o.size()         && !(*o.size()).size())         o.size({});
 if(o.scale_factor() && !(*o.scale_factor()).size()) o.scale_factor({});
 TORCH_CHECK(o.size() || o.scale_factor(), msym(c),": no output size or scale factor given");
 TORCH_CHECK(!(o.size() && o.scale_factor()), msym(c),": both output size and scale factor given");
 return o;
}

static K upsample(bool a,const nn::UpsampleOptions& o) {
 K x=KDICT; nn::UpsampleOptions d;
 if(a || o.size())
  OPTION(x, size, o.size() ? ((*o.size()).size()==1 ? kj((*o.size())[0]) : kget(*o.size())) : ktn(0,0));
 if(a || o.scale_factor())
  OPTION(x, scale, o.scale_factor() ? ((*o.scale_factor()).size()==1 ? kf((*o.scale_factor())[0]) : kget(*o.scale_factor())) : ktn(0,0));
 if(a || o.mode().index() != d.mode().index()) OPTION(x, mode,  ks(ESYM(o.mode())));
 if(a || (d.align_corners() != o.align_corners()) ||
         (d.align_corners() == o.align_corners() &&
          o.align_corners() && *o.align_corners() != *d.align_corners()))
  OPTION(x, align, o.align_corners() ? kb(*o.align_corners()) : ktn(0,0));
 return x;
}

KAPI kinterpolate(K x) {
 KTRY
  TORCH_CHECK(!x->t, "interpolate not implemented for ",kname(x->t));
  Tensor r, *t=xten(x,0);
  return kresult(t,
                 fnn::interpolate(t ? *t : kput(x,0),
                 upsample<fnn::InterpolateFuncOptions>(x,1,Cast::interpolate)));
 KCATCH("interpolate");
}

// --------------------------------------------------------------------------------------
// drop - create dropout module given probability/set dictionary given module
// --------------------------------------------------------------------------------------
static nn::DropoutOptions drop(K x,J i,Cast c) {
 nn::DropoutOptions o; Pairs p; J n=xargc(x,i,p);
 for(J j=0;j<n;++j)
   switch(j) {
    case 0: o.p(mdouble(x,i+j,c,Setting::p)); break;
    case 1: o.inplace(mbool(x,i+j,c,Setting::inplace)); break;
    default: AT_ERROR(msym(c),": up to 2 positional arguments expected, ",n," given");
  }
 while(xpair(p))
  switch(mset(p.k,c)) {
   case Setting::p: o.p(mdouble(p,c)); break;
   case Setting::inplace: o.inplace(mbool(p,c)); break;
   default: AT_ERROR("unrecognized dropout option: ",p.k); break;
  }
 return o;
}

static K drop(bool a,const nn::DropoutOptions& o) {
 K x=KDICT; nn::DropoutOptions d;
 if(a || o.p()       != d.p())       OPTION(x, p,       kf(o.p()));
 if(a || o.inplace() != d.inplace()) OPTION(x, inplace, kb(o.inplace()));
 return x;
}

// --------------------------------------------------------------------------------------
// create embedding/embedding bag module given options:
// embedmode - translate symbol to internal embedding mode (variant)
// embedset - set name/value pairs specific to Embedding vs EmbeddingBag
// embedpair - handle name/value pairs for both types of embedding modules
// embedwt - handle options depending on whether pre-trained weights supplied
// embed, embedbag - process args and return Embedding/EmbeddingBag module
// --------------------------------------------------------------------------------------
static nn::EmbeddingBagMode embedmode(S s) {
 switch(emap(s)) {
  case Enum::sum:  return torch::kSum;
  case Enum::mean: return torch::kMean;
  case Enum::max:  return torch::kMax;
  default: AT_ERROR("unrecognized mode for embedding bag: ",s);
 }
}

static void embedset(Cast c,Setting s,Pairs& p,nn::EmbeddingOptions& o) {
 if(s == Setting::padindex) o.padding_idx(int64n(p,c));
 else AT_ERROR("unrecognized option for ",msym(c),": ",mset(s));
}

static void embedset(Cast c,Setting s,Pairs& p,nn::EmbeddingBagOptions& o) {
 if     (s == Setting::mode)       o.mode(embedmode(code(p,c)));
 else if(s == Setting::lastoffset) o.include_last_offset(mbool(p,c));
 else AT_ERROR("unrecognized option for ",msym(c),": ",mset(s));
}

template<typename O> static void embedpair(Cast c,Pairs& p,O& o,Tensor& w,bool &z) {
 while(xpair(p))
  switch(mset(p.k,c)) {
   case Setting::rows:       o.num_embeddings(int64(p,c)); break;
   case Setting::cols:       o.embedding_dim (int64(p,c)); break;
   case Setting::padindex:   embedset(c,Setting::padindex,p,o); break;
   case Setting::maxnorm:    o.max_norm(optdouble(p,c)); break;
   case Setting::p:          o.norm_type(mdouble(p,c)); break;
   case Setting::scale:      o.scale_grad_by_freq(mbool(p,c)); break;
   case Setting::sparse:     o.sparse(mbool(p,c)); break;
   case Setting::weight:     if(!pempty(p)) pten(p,w); break;
   case Setting::freeze:     z=mbool(p,c); break;
   case Setting::mode:       embedset(c,Setting::mode,p,o); break;
   case Setting::lastoffset: embedset(c,Setting::lastoffset,p,o); break;
   default: AT_ERROR("embedding option: ",p.k," unrecognized");
  }
}

template<typename M,typename O> static M embedwt(Cast c,O o,const Tensor& w,bool z) {
 bool a=o.num_embeddings()!=nj,b=o.embedding_dim()!=nj;
 if(w.defined()) {
  TORCH_CHECK(w.dim()==2, msym(c),": ",w.dim(),"-dim weights given, 2-dim matrix expected");
  TORCH_CHECK(w.is_floating_point(), msym(c),": weight matrix is not floating point");
  if(!a) o.num_embeddings(w.size(0));
  else TORCH_CHECK(o.num_embeddings()==w.size(0), "rows = ",o.num_embeddings()," but weights are ",w.sizes());
  if(!b) o.embedding_dim(w.size(1));
  else TORCH_CHECK(o.embedding_dim() ==w.size(1), "cols = ",o.embedding_dim(), " but weights are ",w.sizes());
   M m=M(o._weight(w));
   m->weight.set_requires_grad(!z);
   return m;
 } else {
  TORCH_CHECK(a,"embed: supply number of rows in the embedding matrix");
  TORCH_CHECK(b,"embed: supply number of cols in the embedding matrix");
  return M(o);
 }
}

static nn::Embedding embed(K x,J i,Cast c) {
 bool z=false; Pairs p; Tensor w; J n=xargc(x,i,p);
 nn::EmbeddingOptions o(nj,nj);
 for(J j=0;j<n;++j) {
   switch(j) {
    case 0:
     switch(kK(x)[i+j]->t) {
      case 0:   if(!xten(x,i+j,w)) w=kput(x,i+j); break;
      case -KJ: o.num_embeddings(int64(x,i+j,c,Setting::rows)); break;
      default:  AT_ERROR("embed: 1st arg is number of rows or weight matrix");
     }
     break;
    case 1: 
     if(w.defined()) z=mbool(x,i+j,c,Setting::freeze);
     else  o.embedding_dim(int64(x,i+j,c,Setting::cols));
     break;
    case 2: o.padding_idx(int64n(x,i+j,c,Setting::padindex)); break;
    case 3: o.max_norm(optdouble(x,i+j,c,Setting::maxnorm)); break;
    case 4: o.norm_type(mdouble(x,i+j,c,Setting::p)); break;
    case 5: o.scale_grad_by_freq(mbool(x,i+j,c,Setting::scale)); break;
    case 6: o.sparse(mbool(x,i+j,c,Setting::sparse)); break;
    default: AT_ERROR(msym(c),": up to 7 positional arguments expected, ",n," given");
  }
 }
 embedpair(c,p,o,w,z);
 return embedwt<nn::Embedding,nn::EmbeddingOptions>(c,o,w,z);
}

static nn::EmbeddingBag embedbag(K x,J i,Cast c) {
 bool z=false; Pairs p; Tensor w; J n=xargc(x,i,p);
 nn::EmbeddingBagOptions o(nj,nj);
 // allow mode if last arg even if early in sequence
 if(!x->t && n>1 && n<6 && xsym(x,i+n-1))
  n--, o.mode(embedmode(code(x,i+n,c,Setting::mode)));
 for(J j=0;j<n;++j) {
   switch(j) {
    case 0:
     switch(kK(x)[i+j]->t) {
      case 0:   if(!xten(x,i+j,w)) w=kput(x,i+j); break;
      case -KJ: o.num_embeddings(int64(x,i+j,c,Setting::rows)); break;
      default:  AT_ERROR("embed: 1st arg is number of rows or weight matrix");
     }
     break;
    case 1: 
     if(w.defined()) z=mbool(x,i+j,c,Setting::freeze);
     else  o.embedding_dim(int64(x,i+j,c,Setting::cols));
     break;
    case 2: o.max_norm(optdouble(x,i+j,c,Setting::maxnorm)); break;
    case 3: o.norm_type(mdouble(x,i+j,c,Setting::p)); break;
    case 4: o.scale_grad_by_freq(mbool(x,i+j,c,Setting::scale)); break;
    case 5: o.mode(embedmode(code(x,i+j,c,Setting::mode))); break;
    case 6: o.sparse(mbool(x,i+j,c,Setting::sparse)); break;
    default: AT_ERROR(msym(c),": up to 7 positional arguments expected, ",n," given");
  }
 }
 embedpair(c,p,o,w,z);
 return embedwt<nn::EmbeddingBag,nn::EmbeddingBagOptions>(c,o,w,z);
}

// -----------------------------------------------------------------------------------------
// retrieve settings from existing Embedding/EmbeddingBag:
// embedget - templated fucntion to retrieve options specific to Embedding or EmbeddingBag
// embed - templated function which gets options and initial optional weights
// -----------------------------------------------------------------------------------------
static void embedget(bool a,K x,Cast c,Setting s,const nn::EmbeddingOptions& o,const nn::EmbeddingOptions& d) {
 if(s == Setting::padindex && (a || o.padding_idx().has_value()))
  OPTION(x, padindex, kj(o.padding_idx() ? o.padding_idx().value() : nj));
}

static void embedget(bool a,K x,Cast c,Setting s,const nn::EmbeddingBagOptions& o,const nn::EmbeddingBagOptions& d) {
 if(s == Setting::mode && (a || o.mode().index() != d.mode().index()))
  OPTION(x, mode, ks(ESYM(o.mode())));
 else if(s == Setting::lastoffset && (a || o.include_last_offset() != d.include_last_offset()))
  OPTION(x, lastoffset, kb(o.include_last_offset()));
}

template<typename O>static K embed(bool a,Cast c,const O& o,const Tensor& w) {
 K x=KDICT; O d(o.num_embeddings(),o.embedding_dim());
 if(o._weight().defined()) {
  OPTION(x, weight, kget(o._weight()));
  OPTION(x, freeze, kb(!w.requires_grad()));
 } else {
  OPTION(x, rows, kj(o.num_embeddings()));
  OPTION(x, cols, kj(o.embedding_dim()));
 }
 embedget(a,x,c,Setting::padindex,o,d); // embedding only
 if(a || o.max_norm().has_value())                         OPTION(x, maxnorm, kf(o.max_norm() ? o.max_norm().value() : nf));
 if(a || o.norm_type()          != d.norm_type())          OPTION(x, p,       kf(o.norm_type()));
 if(a || o.scale_grad_by_freq() != d.scale_grad_by_freq()) OPTION(x, scale,   kb(o.scale_grad_by_freq()));
 embedget(a,x,c,Setting::mode,o,d); //EmbeddingBag only
 if(a || o.sparse()             != d.sparse())             OPTION(x, sparse,  kb(o.sparse()));
 embedget(a,x,c,Setting::lastoffset,o,d);
 return x;
}

// --------------------------------------------------------------------------------------
// linear - parse/retrieve args, invoke functional form
// --------------------------------------------------------------------------------------
static nn::LinearOptions linear(K x,J i,Cast c) {
 bool b=true; int64_t in=nj,out=nj; Pairs p; J n=xargc(x,i,p);
 for(J j=0;j<n;++j) {
   switch(j) {
    case 0:  in=int64(x,i+j,c,Setting::in);   break;
    case 1: out=int64(x,i+j,c,Setting::out);  break;
    case 2:   b=mbool(x,i+j,c,Setting::bias); break;
    default: AT_ERROR(msym(c),": up to 3 positional arguments expected, ",n," given");
  }
 }
 while(xpair(p))
  switch(mset(p.k,c)) {
   case Setting::in:   in=int64(p,c); break;
   case Setting::out:  out=int64(p,c); break;
   case Setting::bias: b=mbool(p,c); break;
   default: AT_ERROR("unrecognized linear option: ",p.k); break;
  }
 TORCH_CHECK(in>0,  msym(c), ": positive input size required");
 TORCH_CHECK(out>0, msym(c), ": positive output size required");
 return nn::LinearOptions(in,out).bias(b);
}

static K linear(bool a,const nn::LinearOptions& o) {
 K x=KDICT; nn::LinearOptions d(o.in_features(),o.out_features());
 OPTION(x, in,  kj(o.in_features()));
 OPTION(x, out, kj(o.out_features()));
 if(a || (o.bias() != d.bias())) OPTION(x, bias, kb(o.bias()));
 return x;
}

KAPI Linear(K x) {
 KTRY
  TORCH_CHECK(!x->t, "linear not implemented for ",kname(x->t));
  TORCH_CHECK(x->n==2 || x->n==3, "linear requires 2-3 args, (input; weight; optional bias)");
  Tensor r, *a=xten(x,0), *w=xten(x,1), *b=xten(x,2);
  if(x->n==2)
   r=torch::linear(a ? *a : kput(x,0), w ? *w : kput(x,1));
  else
   r=torch::linear(a ? *a : kput(x,0), w ? *w : kput(x,1), b ? *b : kput(x,2));
  return kresult(a||w||b, r);
 KCATCH("linear");
}

// --------------------------------------------------------------------------------------
// bilinear - parse/retrieve args, callable functional form
// --------------------------------------------------------------------------------------
static nn::BilinearOptions bilinear(K x,J i,Cast c) {
 bool b=true; int64_t in1=nj,in2=nj,out=nj; Pairs p; J n=xargc(x,i,p);
 for(J j=0;j<n;++j) {
   switch(j) {
    case 0: in1=int64(x,i+j,c,Setting::in1);   break;
    case 1: in2=int64(x,i+j,c,Setting::in2);   break;
    case 2: out=int64(x,i+j,c,Setting::out);  break;
    case 3:   b=mbool(x,i+j,c,Setting::bias); break;
    default: AT_ERROR(msym(c),": up to 4 positional arguments(in1,in2,out,biasflag) expected, ",n," given");
  }
 }
 while(xpair(p))
  switch(mset(p.k,c)) {
   case Setting::in1:  in1=int64(p,c); break;
   case Setting::in2:  in2=int64(p,c); break;
   case Setting::out:  out=int64(p,c); break;
   case Setting::bias: b=mbool(p,c); break;
   default: AT_ERROR("unrecognized bilinear option: ",p.k); break;
  }
 TORCH_CHECK(in1>0 && in2>0, msym(c), ": positive input sizes required");
 TORCH_CHECK(out>0,          msym(c), ": positive output size required");
 return nn::BilinearOptions(in1,in2,out).bias(b);
}

static K bilinear(bool a,const nn::BilinearOptions& o) {
 K x=KDICT; nn::BilinearOptions d(o.in1_features(),o.in2_features(),o.out_features());
 OPTION(x, in1,  kj(o.in1_features()));
 OPTION(x, in2,  kj(o.in2_features()));
 OPTION(x, out, kj(o.out_features()));
 if(a || (o.bias() != d.bias())) OPTION(x, bias, kb(o.bias()));
 return x;
}

KAPI Bilinear(K x) {
 KTRY
  TORCH_CHECK(!x->t, "bilinear not implemented for ",kname(x->t));
  TORCH_CHECK(x->n==3 || x->n==4, "blinear requires 3-4 args, (input1; input2; weight; optional bias)");
  Tensor r, *x1=xten(x,0), *x2=xten(x,1), *w=xten(x,2), *b=xten(x,3);
  return kresult(x1||x2||w||b, torch::bilinear(x1 ? *x1 : kput(x,0),
                                               x2 ? *x2 : kput(x,1),
                                                w ?  *w : kput(x,2),
                                                x->n==3 ? Tensor{} : (b ? *b : kput(x,3))));
 KCATCH("bilinear");
}


// --------------------------------------------------------------------------------------
// rnn - create rnn/gru/lstm module given options/set dictionary of options from module
//     - rnn accepts non-linear function specification: `tanh or `relu
//       gru & lstm don't have that option, so templates/overloading used for fn setting
// --------------------------------------------------------------------------------------
template<typename O> static void rnnfn(O& o,Cast c,S s) {
 AT_ERROR(msym(c),": no non-linear function required (RNN only)");
}

static void rnnfn(nn::RNNOptions& o,Cast c,S s) {
 switch(emap(s)) {
  case Enum::tanh:   o.nonlinearity(torch::kTanh); break;
  case Enum::relu:   o.nonlinearity(torch::kReLU); break;
  default: AT_ERROR("unrecognized RNN fn: ",s); break;
 }
}

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
   default: AT_ERROR(msym(c)," option: ",p.k," unrecognized");
  }
 TORCH_CHECK(o.hidden_size()>0, msym(c), ": hidden size should be greater than zero");
}

static nn::RNNOptions rnn(K x,J i,Cast c) {
 nn::RNNOptions o(0,0); Pairs p; Tensor w; J n=xargc(x,i,p);
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
   default: AT_ERROR(msym(c),": up to 8 positional arguments expected, ",n," given");
  }
 rnnpair(c,p,o);
 return o;
}

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
   default: AT_ERROR(msym(c),": up to 7 positional arguments expected, ",n," given");
  }
 rnnpair(c,p,o);
 return o;
}
 
static S rnnfn(const nn::RNNOptions& o) {return ESYM(o.nonlinearity());}
template<typename O>static S rnnfn(const O& o) {return nullptr;}

template<typename O> static K rnn(bool a,const O& o) {
 K x=KDICT; O d(o.input_size(),o.hidden_size()); S s=rnnfn(o);
 OPTION(x, in,     kj(o.input_size()));
 OPTION(x, hidden, kj(o.hidden_size()));
 if(a || (o.num_layers()    != d.num_layers()))   OPTION(x, layers,     kj(o.num_layers()));
 if((a && s) || s           != rnnfn(d))          OPTION(x, fn,         ks(s));
 if(a || (o.bias()          != d.bias()))         OPTION(x, bias,       kb(o.bias()));
 if(a || (o.batch_first()   != d.batch_first()))  OPTION(x, batchfirst, kb(o.batch_first()));
 if(a || (o.dropout()       != d.dropout()))      OPTION(x, dropout,    kf(o.dropout()));
 if(a || (o.bidirectional() != d.bidirectional()))OPTION(x, bi,         kb(o.bidirectional()));
 return x;
}

// ----------------------------------------------------------------------------------
//  maxpool - process args, return dictionary of options, call functional form
// ----------------------------------------------------------------------------------
template<size_t D> static nn::MaxPoolOptions<D> maxpool(K x,J i,Cast c) {
 nn::MaxPoolOptions<D> o(0);
 bool sz=false,st=false; Pairs p; J n=xargc(x,i,p);
 for(J j=0;j<n;++j) {
   switch(j) {
    case 0: o.kernel_size(exarray<D>(x,i+j,c,Setting::size));    sz=true; break;
    case 1: o.stride     (exarray<D>(x,i+j,c,Setting::stride));  st=true; break;
    case 2: o.padding    (exarray<D>(x,i+j,c,Setting::pad));     break;
    case 3: o.dilation   (exarray<D>(x,i+j,c,Setting::dilate));  break;
    case 4: o.ceil_mode  (mbool     (x,i+j,c,Setting::ceiling)); break;
    default: AT_ERROR(msym(c),": up to 5 positional arguments expected, ",n," given");
  }
 }
 while(xpair(p))
  switch(mset(p.k,c)) {
   case Setting::size:    o.kernel_size(exarray<D>(p,c)); sz=true; break;
   case Setting::stride:  o.stride     (exarray<D>(p,c)); st=true; break;
   case Setting::pad:     o.padding    (exarray<D>(p,c)); break;
   case Setting::dilate:  o.dilation   (exarray<D>(p,c)); break;
   case Setting::ceiling: o.ceil_mode  (mbool(p,c)); break;
   default: AT_ERROR("unrecognized max pooling option: ",p.k); break;
  }
 TORCH_CHECK(sz, msym(c),": no kernel size given");
 if(!st) o.stride(o.kernel_size());
 return o;
}

template<typename O> static K maxpool(bool a,const O& o) {
 K x=KDICT; O d(o.kernel_size());
 OPTION(x, size, KEX(o.kernel_size()));
 if(a || *o.stride()   != *d.stride())   OPTION(x, stride,  KEX(o.stride()));
 if(a || *o.padding()  != *d.padding())  OPTION(x, pad,     KEX(o.padding()));
 if(a || *o.dilation() != *d.dilation()) OPTION(x, dilate,  KEX(o.dilation()));
 if(a || o.ceil_mode() != d.ceil_mode()) OPTION(x, ceiling, kb(o.ceil_mode()));
 return x;
}

// ----------------------------------------------------------------------------------
//  avgpool - process args, return dictionary of options, call functional form
// ----------------------------------------------------------------------------------
template<size_t D> static nn::AvgPoolOptions<D> avgpool(K x,J i,Cast c) {
 nn::AvgPoolOptions<D> o(0);
 bool sz=false,st=false; Pairs p; J n=xargc(x,i,p);
 for(J j=0;j<n;++j) {
   switch(j) {
    case 0: o.kernel_size      (exarray<D>(x,i+j,c,Setting::size));   sz=true; break;
    case 1: o.stride           (exarray<D>(x,i+j,c,Setting::stride)); st=true; break;
    case 2: o.padding          (exarray<D>(x,i+j,c,Setting::pad));      break;
    case 3: o.ceil_mode        (mbool     (x,i+j,c,Setting::ceiling));  break;
    case 4: o.count_include_pad(mbool     (x,i+j,c,Setting::countpad)); break;
    case 5: o.divisor_override (int64n    (x,i+j,c,Setting::divisor));  break;
    default: AT_ERROR(msym(c),": up to 6 positional arguments expected, ",n," given");
  }
 }
 while(xpair(p))
  switch(mset(p.k,c)) {
   case Setting::size:    o.kernel_size (exarray<D>(p,c)); sz=true; break;
   case Setting::stride:  o.stride      (exarray<D>(p,c)); st=true; break;
   case Setting::pad:     o.padding     (exarray<D>(p,c)); break;
   case Setting::ceiling: o.ceil_mode        (mbool(p,c)); break;
   case Setting::countpad:o.count_include_pad(mbool(p,c)); break;
   case Setting::divisor: o.divisor_override(int64n(p,c)); break;
   default: AT_ERROR("unrecognized avg pooling option: ",p.k); break;
  }
 TORCH_CHECK(sz, msym(c),": no kernel size given");
 if(!st) o.stride(o.kernel_size());
 return o;
}

template<typename O> static K avgpool(bool a,const O& o) {
 K x=KDICT; O d(o.kernel_size());
 OPTION(x, size, KEX(o.kernel_size()));
 if(a || *o.stride()           != *d.stride())           OPTION(x, stride,   KEX(o.stride()));
 if(a || *o.padding()          != *d.padding())          OPTION(x, pad,      KEX(o.padding()));
 if(a || o.ceil_mode()         != d.ceil_mode())         OPTION(x, ceiling,  kb(o.ceil_mode()));
 if(a || o.count_include_pad() != d.count_include_pad()) OPTION(x, countpad, kb(o.count_include_pad()));
 if(a || o.divisor_override().has_value())               OPTION(x, divisor,  kj(o.divisor_override() ? o.divisor_override().value() : nj));
 return x;
}

// ---------------------------------------------------------------------------------------
// adaptive pooling - process args, return dictionary of options, call functional form
// adapt - multiple versions to handle expanding array(1d) vs array of optionals(2,3d)
// ---------------------------------------------------------------------------------------
template<size_t D> static void adapt(ExpandingArray<D>& a,K x,J i,Cast c)        {a=exarray<D>(x,i,c,Setting::size);}
template<size_t D> static void adapt(ExpandingArray<D>& a,const Pairs& p,Cast c) {a=exarray<D>(p,c);}
template<size_t D> static void adapt(Exoptional<D>& a,K x,J i,Cast c)        {a=exoptional<D>(x,i,c,Setting::size);}
template<size_t D> static void adapt(Exoptional<D>& a,const Pairs& p,Cast c) {a=exoptional<D>(p,c);}

template<size_t D> static bool adapt(ExpandingArray<D>& a) {for(auto &v:*a) if(v != nj) return true; return false;}
template<size_t D> static bool adapt(Exoptional<D>& a)     {for(auto &v:*a) if(v)       return true; return false;}

template<size_t D,typename T> static T adapt(K x,J i,Cast c) {
 T o(0); bool sz=false; Pairs p; J n=xargc(x,i,p);
 for(J j=0;j<n;++j) {
   switch(j) {
    case 0: adapt<D>(o.output_size(),x,i+j,c); sz=true; break;
    default: AT_ERROR(msym(c),": 1 positional argument expected, ",n," given");
  }
 }
 while(xpair(p))
  switch(mset(p.k,c)) {
   case Setting::size: adapt<D>(o.output_size(),p,c); sz=true; break;
   default: AT_ERROR("unrecognized ",msym(c)," option: ",p.k); break;
  }
 TORCH_CHECK(sz, msym(c),": no output size given");
 TORCH_CHECK(adapt(o.output_size()), msym(c),": no output size");
 return o;
}

template<typename O> static K adapt(const O& o) {
 K x=KDICT;
 OPTION(x, size, KEX(o.output_size()));
 return x;
}

// ----------------------------------------------------------------------------------
// fpool - fractional max pooling for 2 & 3d layers
// ----------------------------------------------------------------------------------
template<size_t D> static nn::FractionalMaxPoolOptions<D> fpool(K x,J i,Cast c) {
 nn::FractionalMaxPoolOptions<D> o(0);
 bool e,sz=false; Pairs p; J n=xargc(x,i,p);
 for(J j=0;j<n;++j) {
   e=xempty(x,i+j);
   switch(j) {
    case 0: o.kernel_size(exarray<D>(x,i+j,c,Setting::size)); sz=true; break;
    case 1: if(e) o.output_size( c10::nullopt); else o.output_size ( exarray<D>(x,i+j,c,Setting::outsize)); break;
    case 2: if(e) o.output_ratio(c10::nullopt); else o.output_ratio(exdouble<D>(x,i+j,c,Setting::ratio));   break;
    default: AT_ERROR(msym(c),": up to 3 positional arguments expected, ",n," given");
  }
 }
 while(xpair(p)) {
  e=pempty(p);
  switch(mset(p.k,c)) {
   case Setting::size:    o.kernel_size(exarray<D>(p,c)); sz=true; break;
   case Setting::outsize: if(e) o.output_size (c10::nullopt); else o.output_size(exarray  <D>(p,c)); break;
   case Setting::ratio:   if(e) o.output_ratio(c10::nullopt); else o.output_ratio(exdouble<D>(p,c)); break;
   default: AT_ERROR("unrecognized ",msym(c)," option: ",p.k); break;
  }
 }
 TORCH_CHECK(sz, msym(c), ": no kernel size given");
 TORCH_CHECK(o.output_size()||o.output_ratio(), msym(c), ": no output size or ratio given");
 TORCH_CHECK(!(o.output_size()&&o.output_ratio()), msym(c), ": cannot specify both output size & output ratio");
 return o;
}

template<typename O> static K fpool(bool a,const O& o) {
 K x=KDICT;
 OPTION(x, size, KEX(o.kernel_size()));
 if(a || o.output_size().has_value())    OPTION(x, outsize, o.output_size() ? KEX(o.output_size().value())  : ktn(0,0));
 if(a || o.output_ratio().has_value())   OPTION(x, ratio,   o.output_ratio()? KEX(o.output_ratio().value()) : ktn(0,0));
 return x;
}

// ----------------------------------------------------------------------------------
// lppool - power-average pooling
// ----------------------------------------------------------------------------------
template<size_t D> static nn::LPPoolOptions<D> lppool(K x,J i,Cast c) {
 nn::LPPoolOptions<D> o(0,0);
 bool pw=false,sz=false,st=false; Pairs p; J n=xargc(x,i,p);
 for(J j=0;j<n;++j) {
   switch(j) {
    case 0: o.norm_type  (mdouble(x,i+j,c,Setting::p));         pw=true; break;
    case 1: o.kernel_size(exarray<D>(x,i+j,c,Setting::size));   sz=true; break;
    case 2: o.stride     (exarray<D>(x,i+j,c,Setting::stride)); st=true; break;
    case 3: o.ceil_mode  (mbool    (x,i+j,c,Setting::ceiling)); break;
    default: AT_ERROR(msym(c),": up to 4 positional arguments expected, ",n," given");
  }
 }
 while(xpair(p))
  switch(mset(p.k,c)) {
   case Setting::p:       o.norm_type  (mdouble   (p,c)); pw=true; break;
   case Setting::size:    o.kernel_size(exarray<D>(p,c)); sz=true; break;
   case Setting::stride:  o.stride     (exarray<D>(p,c)); st=true; break;
   case Setting::ceiling: o.ceil_mode  (mbool(p,c)); break;
   default: AT_ERROR("unrecognized ",msym(c)," option: ",p.k); break;
  }
 TORCH_CHECK(pw, msym(c),": no power given");
 TORCH_CHECK(sz, msym(c),": no kernel size given");
 if(!st) o.stride(o.kernel_size());
 return o;
}

template<typename O> static K lppool(bool a,const O& o) {
 K x=KDICT; O d(o.norm_type(),o.kernel_size());
 OPTION(x, p,    kf(o.norm_type()));
 OPTION(x, size, KEX(o.kernel_size()));
 if(a || *o.stride()   != *d.stride())   OPTION(x, stride,  KEX(o.stride()));
 if(a || o.ceil_mode() != d.ceil_mode()) OPTION(x, ceiling, kb(o.ceil_mode()));
 return x;
}

// ----------------------------------------------------------------------------------
// functional form of pooling methods:
// ----------------------------------------------------------------------------------
static K pool(K x,Cast c) {
 KTRY
  TORCH_CHECK(!x->t, msym(c)," not implemented for ",kname(x->t));
  Tensor r, *t=xten(x,0);
  switch(c) {
   case Cast::maxpool1d:  r=fnn::max_pool1d(t ? *t : kput(x,0), maxpool<1>(x,1,c)); break;
   case Cast::maxpool2d:  r=fnn::max_pool2d(t ? *t : kput(x,0), maxpool<2>(x,1,c)); break;
   case Cast::maxpool3d:  r=fnn::max_pool3d(t ? *t : kput(x,0), maxpool<3>(x,1,c)); break;
   case Cast::avgpool1d:  r=fnn::avg_pool1d(t ? *t : kput(x,0), avgpool<1>(x,1,c)); break;
   case Cast::avgpool2d:  r=fnn::avg_pool2d(t ? *t : kput(x,0), avgpool<2>(x,1,c)); break;
   case Cast::avgpool3d:  r=fnn::avg_pool3d(t ? *t : kput(x,0), avgpool<3>(x,1,c)); break;
   case Cast::adaptmax1d: r=fnn::adaptive_max_pool1d(t ? *t : kput(x,0), adapt<1,nn::AdaptiveMaxPool1dOptions>(x,1,c)); break;
   case Cast::adaptmax2d: r=fnn::adaptive_max_pool2d(t ? *t : kput(x,0), adapt<2,nn::AdaptiveMaxPool2dOptions>(x,1,c)); break;
   case Cast::adaptmax3d: r=fnn::adaptive_max_pool3d(t ? *t : kput(x,0), adapt<3,nn::AdaptiveMaxPool3dOptions>(x,1,c)); break;
   case Cast::adaptavg1d: r=fnn::adaptive_avg_pool1d(t ? *t : kput(x,0), adapt<1,nn::AdaptiveAvgPool1dOptions>(x,1,c)); break;
   case Cast::adaptavg2d: r=fnn::adaptive_avg_pool2d(t ? *t : kput(x,0), adapt<2,nn::AdaptiveAvgPool2dOptions>(x,1,c)); break;
   case Cast::adaptavg3d: r=fnn::adaptive_avg_pool3d(t ? *t : kput(x,0), adapt<3,nn::AdaptiveAvgPool3dOptions>(x,1,c)); break;
   case Cast::fmaxpool2d: r=fnn::fractional_max_pool2d(t ? *t : kput(x,0), fpool<2>(x,1,c)); break;
   case Cast::fmaxpool3d: r=fnn::fractional_max_pool3d(t ? *t : kput(x,0), fpool<3>(x,1,c)); break;
   case Cast::lppool1d:   r=fnn::lp_pool1d(t ? *t : kput(x,0), lppool<1>(x,1,c)); break;
   case Cast::lppool2d:   r=fnn::lp_pool2d(t ? *t : kput(x,0), lppool<2>(x,1,c)); break;
   default: AT_ERROR("unrecognized pooling function");
  }
  return kresult(t,r);
 KCATCH("pool");
}

KAPI maxpool1d(K x)  {return pool(x,Cast::maxpool1d);}
KAPI maxpool2d(K x)  {return pool(x,Cast::maxpool2d);}
KAPI maxpool3d(K x)  {return pool(x,Cast::maxpool3d);}
KAPI avgpool1d(K x)  {return pool(x,Cast::avgpool1d);}
KAPI avgpool2d(K x)  {return pool(x,Cast::avgpool2d);}
KAPI avgpool3d(K x)  {return pool(x,Cast::avgpool3d);}
KAPI adaptmax1d(K x) {return pool(x,Cast::adaptmax1d);}
KAPI adaptmax2d(K x) {return pool(x,Cast::adaptmax2d);}
KAPI adaptmax3d(K x) {return pool(x,Cast::adaptmax3d);}
KAPI adaptavg1d(K x) {return pool(x,Cast::adaptavg1d);}
KAPI adaptavg2d(K x) {return pool(x,Cast::adaptavg2d);}
KAPI adaptavg3d(K x) {return pool(x,Cast::adaptavg3d);}
KAPI fmaxpool2d(K x) {return pool(x,Cast::fmaxpool2d);}
KAPI fmaxpool3d(K x) {return pool(x,Cast::fmaxpool3d);}
KAPI lppool1d(K x)   {return pool(x,Cast::lppool1d);}
KAPI lppool2d(K x)   {return pool(x,Cast::lppool2d);}

// ----------------------------------------------------------------------------------
// padmode - match k symbol to std::variant style enumeration
// pad - n-dimensional padding, specify even number of sizes and optional pad value
// ----------------------------------------------------------------------------------
static void padmode(fnn::PadFuncOptions& o,S s) {
 switch(emap(s)) {
  case Enum::constant:  o.mode(torch::kConstant); break;
  case Enum::reflect:   o.mode(torch::kReflect); break;
  case Enum::replicate: o.mode(torch::kReplicate); break;
  case Enum::circular:  o.mode(torch::kCircular); break;
  default: AT_ERROR("unrecognized padding mode: ",s); break;
 }
}

static fnn::PadFuncOptions pad(K x,J i,Cast c) {
 fnn::PadFuncOptions o({}); S s; Pairs p; J n=xargc(x,i,p);
 for(J j=0;j<n;++j)
  switch(j) {
   case 0: o.pad(mlongs(x,i+j,c,Setting::pad)); break;
   case 1:
    if(xsym(x,i+j,s)) padmode(o,s);
    else if(n==2)     o.value(mdouble(x,i+j,c,Setting::value));
    else AT_ERROR("pad: unrecognized 2nd arg, expecting mode or value");
    break;
   case 2: o.value(mdouble(x,i+j,c,Setting::value)); break;
   default: AT_ERROR(msym(c),": up to 3 positional args expected(padding;mode;value), ",n," given");
  }
 while(xpair(p))
  switch(mset(p.k,c)) {
   case Setting::pad:   o.pad(mlongs(p,c)); break;
   case Setting::mode:  padmode(o,code(p,c)); break;
   case Setting::value: o.value(mdouble(p,c)); break;
   default: AT_ERROR("padding option: ",p.k," not recognized");
  }
 n=o.pad().size();
 TORCH_CHECK(n>0 && !(n % 2), msym(c),": ",n," pad size(s) given, expecting pairs for left,right or left,right,top,bottom.. etc");
 return o;
}

static K pad(bool a,const PadImpl* m) {
 K x=KDICT; const fnn::PadFuncOptions d({}), &o=m->options;
 OPTION(x, pad, klist(o.pad().size(),o.pad().data()));
 if(a || o.mode().index() != d.mode().index()) OPTION(x, mode,  ks(ESYM(o.mode())));
 if(a || o.value()        != d.value())        OPTION(x, value, kf(o.value()));
 return x;
}

KAPI kpad(K x) {
 KTRY
  TORCH_CHECK(!x->t, "pad not implemented for ",kname(x->t));
  Tensor r, *t=xten(x,0);
  return kresult(t, fnn::pad(t ? *t : kput(x,0), pad(x,1,Cast::pad)));
 KCATCH("pad");
}

// ----------------------------------------------------------------------------------
// cpad - constant pad w'fixed dimension and optional value (defaults to zero)
// ----------------------------------------------------------------------------------
template<size_t D,typename M> static M cpad(K x,J i,Cast c) {
 M o(0,0);
 bool sz=false; Pairs p; J n=xargc(x,i,p);
 for(J j=0;j<n;++j)
   switch(j) {
    case 0: o.padding(exarray<D*2>(x,i+j,c,Setting::pad)); sz=true; break;
    case 1: o.value(mdouble(x,i+j,c,Setting::value)); break;
    default: AT_ERROR(msym(c),": up to 2 positional args expected(padding;value), ",n," given");
  }
 while(xpair(p))
  switch(mset(p.k,c)) {
   case Setting::pad: o.padding(exarray<D*2>(p,c)); sz=true; break;
   case Setting::value: o.value(mdouble(p,c)); break;
   default: AT_ERROR("unrecognized ",msym(c)," option: ",p.k); break;
  }
 TORCH_CHECK(sz, msym(c),": no padding sizes given");
 return o;
}

template<typename O> static K cpad(const O& o) {
 K x=KDICT;
 OPTION(x, pad, KEX(o.padding()));
 OPTION(x, value, kf(o.value()));
 return x;
}

// ----------------------------------------------------------------------------------
// npad - reflect/replicate/zero pad w'fixed dimension
// ----------------------------------------------------------------------------------
template<size_t D,typename M> static M npad(K x,J i,Cast c) {
 M o(0);
 bool sz=false; Pairs p; J n=xargc(x,i,p);
 for(J j=0;j<n;++j)
   switch(j) {
    case 0: o.padding(exarray<D*2>(x,i+j,c,Setting::pad)); sz=true; break;
    default: AT_ERROR(msym(c),": only 1 positional argument expected, ",n," given");
  }
 while(xpair(p))
  switch(mset(p.k,c)) {
   case Setting::pad: o.padding(exarray<D*2>(p,c)); sz=true; break;
   default: AT_ERROR("unrecognized ",msym(c)," option: ",p.k); break;
  }
 TORCH_CHECK(sz, msym(c),": no padding sizes given");
 return o;
}

template<typename M> static K npad(const M* m) {
 K x=KDICT;
 OPTION(x, pad, KEX(m->options.padding()));
 return x;
}

// ------------------------------------------------------------------------------------
// noarg:  activation fns w'out args, logsigmoid,sigmoid,softsign,tanh,tanhshrink
// ------------------------------------------------------------------------------------
static void noarg(Cast c,K x,J i) {TORCH_CHECK(xnone(x,i), msym(c), ": no arguments expected, ", kstring(x));}

using Ft = Tensor (*)(const Tensor&);
static K noarg(const char* s,Ft f, K x) {
 KTRY
  Tensor *t=xten(x); return kresult(t, f(t ? *t : kput(x)));
 KCATCH(s);
}

KAPI gelu(K x)       {return noarg("gelu",       torch::gelu,                       x);}
KAPI logsigmoid(K x) {return noarg("logsigmoid", torch::log_sigmoid,                x);}
KAPI softsign(K x)   {return noarg("softsign",   fnn::softsign,   x);}
KAPI tanhshrink(K x) {return noarg("tanhshrink", fnn::tanhshrink, x);}

// ------------------------------------------------------------------------------------
// activation fns with inplace flag as only arg: relu,relu6,selu
// ------------------------------------------------------------------------------------
static bool inplace(K x,J i,Cast c) {
 bool b=false; Pairs p; J n=xargc(x,i,p);
 if(n)
  TORCH_CHECK(xbool(x,i,b) && n==1, msym(c),": unrecognized option(s), expecting single boolean flag");
 while(xpair(p)) {
  TORCH_CHECK(mset(p.k,c)==Setting::inplace, msym(c),": unrecognized option: ",p.k);
  b=mbool(p,c);
 }
 return b;
}

static K inplace(bool a,bool b) {K x=KDICT; if(a || b) OPTION(x, inplace, kb(b)); return x;}

// ------------------------------------------------------------------------------------
//  elu,celu - exponential & continuously differentiable linear unit
//             accepts optional alpha & inplace flag
// ------------------------------------------------------------------------------------
template<typename O> static O alpha(K x,J i,Cast c) {
 O o; Pairs p; bool b; J n=xargc(x,i,p);
 if(n==1) {
  if(xbool(x,i,b))
    o.inplace(b);
  else
    o.alpha(mdouble(x,i,c,Setting::alpha));
 } else if(n==2) {
   o.alpha(mdouble(x,i,   c, Setting::alpha));
   o.inplace(mbool(x,i+1, c, Setting::inplace));
 } else if(n) {
  AT_ERROR(msym(c), ": unrecognized positional option(s), expecting alpha, inplace flag, or (alpha;inplace flag)");
 }
 while(xpair(p))
  switch(mset(p.k,c)) {
   case Setting::alpha:   o.alpha(mdouble(p,c)); break;
   case Setting::inplace: o.inplace(mbool(p,c)); break;
   default: AT_ERROR(msym(c)," option: ",p.k," not recognized");
  }
 return o;
}

template<typename O>static K alpha(bool a,const O& o) {
 K x=KDICT; O d;
 if(a || o.alpha()   != d.alpha())   OPTION(x, alpha,   kf(o.alpha()));
 if(a || o.inplace() != d.inplace()) OPTION(x, inplace, kb(o.inplace()));
 return x;
}

// ------------------------------------------------------------------------------------
//  leakyrelu - allow a small positive gradient(slope) when x<0
// ------------------------------------------------------------------------------------
static nn::LeakyReLUOptions slope(K x,J i,Cast c) {
 nn::LeakyReLUOptions o; Pairs p; bool b; J n=xargc(x,i,p);
 if(n==1) {
  if(xbool(x,i,b))
    o.inplace(b);
  else
    o.negative_slope(mdouble(x,i,c,Setting::slope));
 } else if(n==2) {
   o.negative_slope(mdouble(x, i, c, Setting::slope));
   o.inplace(mbool(x,i+1, c, Setting::inplace));
 } else if(n) {
  AT_ERROR(msym(c), ": unrecognized positional option(s), expecting slope, inplace flag, or (slope;inplace flag)");
 }
 while(xpair(p))
  switch(mset(p.k,c)) {
   case Setting::slope:   o.negative_slope(mdouble(p,c)); break;
   case Setting::inplace: o.inplace(mbool(p,c)); break;
   default: AT_ERROR(msym(c)," option: ",p.k," not recognized");
  }
 return o;
}

static K slope(bool a,Cast c,const nn::LeakyReLUOptions& o) {
 K x=KDICT; nn::LeakyReLUOptions d;
 if(a || o.negative_slope()   != d.negative_slope()) OPTION(x, slope,   kf(o.negative_slope()));
 if(a || o.inplace()          != d.inplace())        OPTION(x, inplace, kb(o.inplace()));
 return x;
}

// ------------------------------------------------------------------------------------
// hardshrink, softshrink - module/function requires single parm: lambda
// ------------------------------------------------------------------------------------
static double lambda(Cast c) {
 return c==Cast::hardshrink ? nn::HardshrinkOptions().lambda() 
                            : nn::SoftshrinkOptions().lambda();
}

static double lambda(K x,J i,Cast c) {
 double l=lambda(c); Pairs p; J n=xargc(x,i,p);
 if(n==1) l=mdouble(x,i,c,Setting::lambda);
 TORCH_CHECK(n<2,msym(c),": unrecognized positional option(s), expecting lambda, e.g. 0.5");
 while(xpair(p)) {
  TORCH_CHECK(mset(p.k,c)==Setting::lambda,"unrecognized option: ",p.k); l=mdouble(p,c);
 }
 return l;
}

static K lambda(bool a,Cast c,double l) {
 K x=KDICT;
 if(a || l != lambda(c)) OPTION(x,lambda,kf(l));
 return x;
}

// ------------------------------------------------------------------------------------
// cat, glu & softmax,softmax,logsoftmax (modules only) accept single dimension arg
// ------------------------------------------------------------------------------------
static int64_t dim(Cast c) {
 switch(c) {
  case Cast::glu: return nn::GLUOptions().dim();
  case Cast::cat: return CatOptions().dim();
  default:        return nj;
 }
}

static int64_t dim(K x,J i,Cast c) {
 int64_t d=dim(c); Pairs p; J n=xargc(x,i,p);
 if(n==1) d=int64(x,i,c,Setting::dim);
 TORCH_CHECK(n<2, msym(c),": unrecognized positional option(s), expecting single dimension");
 while(xpair(p)) {
  TORCH_CHECK(mset(p.k,c)==Setting::dim,"unrecognized option: ",p.k); d=int64(p,c);
 }
 TORCH_CHECK(d!=nj, msym(c),": no dimension given");
 return d;
}

static K dim(bool a,Cast c,int64_t d) {
 K x=KDICT;
 if(a || d != dim(c)) OPTION(x,dim,kj(d));
 return x;}

// ----------------------------------------------------------------------------------
// softmax,softmin,logsoftmax: functional form requires dim & optional data type
// softdim: get default dimension from input tensor dimensions (deprecated)
// ----------------------------------------------------------------------------------
static J softdim(size_t d) {return !(d==0 || d==1 || d==3);}

static void softargs(K x,J i,Cast c,J &d,c10::optional<ScalarType>& s) { 
 s=c10::nullopt; Pairs p; J n=xargc(x,i,p);
 if(!((n==0 && p.n) || (xlong(x,i,d) && (n==1 || (n==2 && xtype(x,i+1,s))))))
  AT_ERROR(msym(c),": unrecognized arg(s), expecting dim or (dim;data type)");
 while(xpair(p))
  switch(mset(p.k,c)) {
   case Setting::dim:  d=plong(p); break;
   case Setting::type: s=ptype(p); break;
   default: AT_ERROR("unrecognized ",msym(c)," option: ",p.k); break;
  }
 if(null(d)) 
  AT_ERROR("specify the dimension along which ",msym(c)," will be computed");
}

// -----------------------------------------------------------------------------------
// rrelu - randomized leaky relu, functional form has an additional flag for training
// -----------------------------------------------------------------------------------
static void rrelu(K x,J i,Cast c,bool fn,bool& tr,bool& in,double& lo,double& up) {
 Pairs p; J n=xargc(x,i,p); fnn::RReLUFuncOptions o;
 lo=o.lower(); up=o.upper(); in=o.inplace(); tr=o.training();
 if(n) {
  if(fn) {
   TORCH_CHECK((n==1 && (xnum(x,i,lo) || xbool(x,i,tr))) ||
               (n==2 &&  xnum(x,i,lo) && (xnum(x,i+1,up) || xbool(x,i+1,tr))) ||
               (n==3 &&  xnum(x,i,lo) &&  xnum(x,i+1,up) && xbool(x,i+2,tr))  ||
               (n==4 &&  xnum(x,i,lo) &&  xnum(x,i+1,up) && xbool(x,i+2,tr) && xbool(x,i+3,in)),
               "rrelu: unexpected positional arg(s), expects (lower;upper;train flag;inplace flag)");
  } else {
   TORCH_CHECK((n==1 && (xnum(x,i,lo) || xbool(x,i,in))) ||
               (n==2 &&  xnum(x,i,lo) && (xnum(x,i+1,up) || xbool(x,i+1,in))) ||
               (n==3 &&  xnum(x,i,lo) &&  xnum(x,i+1,up) && xbool(x,i+2,in)),
               "rrelu: unexpected positional arg(s), expects (lower;upper;inplace flag)");
  }
 }
 while(xpair(p))
  switch(mset(p.k,c)) {
   case Setting::lower:   lo=mdouble(p,c); break;
   case Setting::upper:   up=mdouble(p,c); break;
   case Setting::train:   TORCH_CHECK(fn,"rrelu: training flag not set for module"); tr=mbool(p,c);   break;
   case Setting::inplace: in=mbool(p,c);   break;
   default: AT_ERROR("rrelu option: ",p.k," not recognized");
  }
}

// return options for rrelu module
static nn::RReLUOptions rrelu(K x,J i,Cast c) {
 double lo,up; bool in,tr; rrelu(x,i,c,false,tr,in,lo,up);
 return nn::RReLUOptions().lower(lo).upper(up).inplace(in);
}

// retrieve options from rrelu module
static K rrelu(bool a,const nn::RReLUOptions& o) {
 K x=KDICT; nn::RReLUOptions d;
 if(a || d.lower()   != o.lower())   OPTION(x, lower,   kf(o.lower()));
 if(a || d.upper()   != o.upper())   OPTION(x, upper,   kf(o.upper()));
 if(a || d.inplace() != o.inplace()) OPTION(x, inplace, kb(o.inplace()));
 return x;
}

// -----------------------------------------------------------------------------------------
// hardtanh - computationally cheaper version of tanh, straight line at min,max
// -----------------------------------------------------------------------------------------
static nn::HardtanhOptions hardtanh(K x,J i,Cast c) {
 Pairs p; J n=xargc(x,i,p); nn::HardtanhOptions o;
 bool b=o.inplace(); double v1=o.min_val(),v2=o.max_val();
 if(n) {
  TORCH_CHECK((n==1 && (xnum(x,i,v1) || xbool(x,i,b))) ||
              (n==2 &&  xnum(x,i,v1) && (xnum(x,i+1,v2) || xbool(x,i+1,b))) ||
              (n==3 &&  xnum(x,i,v1) &&  xnum(x,i+1,v2) && xbool(x,i+2,b)),
              "hardtanh: unexpected positional arg(s), expects (min;max;inplace flag)");
 }
 while(xpair(p))
  switch(mset(p.k,c)) {
   case Setting::min:     v1=mdouble(p,c); break;
   case Setting::max:     v2=mdouble(p,c); break;
   case Setting::inplace: b=mbool(p,c); break;
   default: AT_ERROR("hardtanh option: ",p.k," not recognized");
  }
 return o.min_val(v1).max_val(v2).inplace(b);
}

static K hardtanh(bool a,const nn::HardtanhOptions& o) {
 K x=KDICT; nn::HardtanhOptions d;
 if(a || d.min_val() != o.min_val()) OPTION(x, min,     kf(o.min_val()));
 if(a || d.max_val() != o.max_val()) OPTION(x, max,     kf(o.max_val()));
 if(a || d.inplace() != o.inplace()) OPTION(x, inplace, kb(o.inplace()));
 return x;
}

// -----------------------------------------------------------------------------------------
// softplus - smooth approximation to relu, can constrain to always be positive
// -----------------------------------------------------------------------------------------
static nn::SoftplusOptions softplus(K x,J i,Cast c) {
 Pairs p; J n=xargc(x,i,p); nn::SoftplusOptions o; double v1=o.beta(),v2=o.threshold();
 if(n) {
  TORCH_CHECK(xnum(x,i,v1) && (n==1 || (n==2 && xnum(x,i+1,v2))),
              "softplus: unexpected positional arg(s), expects (beta;threshold)");
 }
 while(xpair(p))
  switch(mset(p.k,c)) {
   case Setting::beta:      v1=mdouble(p,c); break;
   case Setting::threshold: v2=mdouble(p,c); break;
   default: AT_ERROR("softplus option: ",p.k," not recognized");
  }
 return o.beta(v1).threshold(v2);
}

static K softplus(bool a,const nn::SoftplusOptions& o) {
 K x=KDICT; nn::SoftplusOptions d;
 if(a || d.beta()      != o.beta())      OPTION(x, beta,      kf(o.beta()));
 if(a || d.threshold() != o.threshold()) OPTION(x, threshold, kf(o.threshold()));
 return x;
}

// ----------------------------------------------------------------------------------------------
// threshold - thresholds each element of input tensor, fns set/get threshold,value,inplace flag
// ----------------------------------------------------------------------------------------------
static nn::ThresholdOptions threshold(K x,J i,Cast c) {
 Pairs p; J n=xargc(x,i,p); bool b=false; double v1=nf,v2=nf;
 if(n) {
  TORCH_CHECK((n==1 && (xnum(x,i,v1) || xbool(x,i,b))) ||
              (n==2 &&  xnum(x,i,v1) && (xnum(x,i+1,v2) || xbool(x,i+1,b))) ||
              (n==3 &&  xnum(x,i,v1) &&  xnum(x,i+1,v2) && xbool(x,i+2,b)),
              "threshold: unexpected positional arg(s), expects (threshold;value;inplace flag)");
 }
 while(xpair(p))
  switch(mset(p.k,c)) {
   case Setting::threshold: v1=mdouble(p,c); break;
   case Setting::value:     v2=mdouble(p,c); break;
   case Setting::inplace:   b=mbool(p,c); break;
   default: AT_ERROR("threshold option: ",p.k," not recognized");
  }
 TORCH_CHECK(v1 == v1 && v2 == v2, "threshold: both threshold level & replacement value must be given");
 return nn::ThresholdOptions(v1,v2).inplace(b);
}

static K threshold(bool a,const nn::ThresholdOptions& o) {
 K x=KDICT;
 OPTION(x, threshold, kf(o.threshold()));
 OPTION(x, value,     kf(o.value()));
 if(a || o.inplace()) OPTION(x, inplace, kb(o.inplace()));
 return x;
}

// -----------------------------------------------------------------------------------------
// functional form of activation fns:
//  relu,relu6,selu (inplace flag), elu,celu(alpha & inplace), leakyrelu(slope & inplace),
//  hardshrink,softshrink(lambda), glu(dim), rrelu(lower,upper & inplace flag)
//  hardtanh(min,max,inplace), softplus(beta,threshold), threshold(threshold,value,inplace)
// -----------------------------------------------------------------------------------------
static K act(K x,Cast c,const char* s) {
 KTRY
  bool a,p; Tensor r,t;
  if(xten(x,t))        p=true, a=false;
  else if(xten(x,0,t)) p=true, a=true;
  else if(xmixed(x,3)) p=false,a=true, t=kput(x,0);
  else                 p=false,a=false,t=kput(x);
  switch(c) {
   case Cast::relu:  r=fnn::relu (t,a ? inplace(x,1,c) : false); break;
   case Cast::relu6: r=fnn::relu6(t,a ? inplace(x,1,c) : false); break;
   case Cast::selu:  r=fnn::selu (t,a ? inplace(x,1,c) : false); break;
   case Cast::elu:   r=fnn::elu(t,alpha<nn::ELUOptions>(a ? x : nullptr,1,c)); break;
   case Cast::celu:  r=fnn::celu(t,alpha<nn::CELUOptions>(a ? x : nullptr,1,c)); break;
   case Cast::leakyrelu: r=fnn::leaky_relu(t,slope(a ? x : nullptr,1,c)); break;
   case Cast::hardshrink: r=torch::hardshrink(t,a ? lambda(x,1,c) : lambda(c)); break;
   case Cast::softshrink: r=torch::softshrink(t,a ? lambda(x,1,c) : lambda(c)); break;
   case Cast::glu:        r=fnn::glu(t,a ? dim(x,1,c) : dim(c)); break;
   case Cast::softmin:
   case Cast::softmax:
   case Cast::logsoftmax: {
    auto d=softdim(t.dim()); c10::optional<ScalarType> s; if(a) softargs(x,1,c,d,s);
    switch(c) {
     case Cast::softmin:    r=fnn::detail::softmin(t,d,s); break;
     case Cast::softmax:    r=fnn::detail::softmax(t,d,s); break;
     case Cast::logsoftmax: r=fnn::detail::log_softmax(t,d,s); break;
     default: AT_ERROR("unrecognized activation function");
    }
    break;
   }
   case Cast::rrelu: {
    double lo,up; bool in,tr; rrelu(a ? x : nullptr,1,c,false,tr,in,lo,up);
    r=fnn::detail::rrelu(t,lo,up,tr,in);
    break;
   }
   case Cast::hardtanh:  r=fnn::hardtanh (t,  hardtanh(a ? x : nullptr,1,c)); break;
   case Cast::softplus:  r=fnn::softplus (t,  softplus(a ? x : nullptr,1,c)); break;
   case Cast::threshold: r=fnn::threshold(t, threshold(a ? x : nullptr,1,c)); break;
   default: AT_ERROR("unrecognized activation function"); break;
  }
  return p && r.is_same(t) ? (K)0 : kresult(p,r);
 KCATCH(s);
}

KAPI       relu(K x) {return act(x, Cast::relu,       "relu");}
KAPI      relu6(K x) {return act(x, Cast::relu6,      "relu6");}
KAPI       selu(K x) {return act(x, Cast::selu,       "selu");}
KAPI        elu(K x) {return act(x, Cast::elu,        "elu");}
KAPI       celu(K x) {return act(x, Cast::celu,       "celu");}
KAPI  leakyrelu(K x) {return act(x, Cast::leakyrelu,  "leakyrelu");}
KAPI hardshrink(K x) {return act(x, Cast::hardshrink, "hardshrink");}
KAPI softshrink(K x) {return act(x, Cast::softshrink, "softshrink");}
KAPI        glu(K x) {return act(x, Cast::glu,        "glu");}
KAPI    softmin(K x) {return act(x, Cast::softmin,    "softmin");}
KAPI    softmax(K x) {return act(x, Cast::softmax,    "softmax");}
KAPI logsoftmax(K x) {return act(x, Cast::logsoftmax, "logsoftmax");}
KAPI      Rrelu(K x) {return act(x, Cast::rrelu,      "rrelu");}
KAPI   Hardtanh(K x) {return act(x, Cast::hardtanh,   "hardtanh");}
KAPI   Softplus(K x) {return act(x, Cast::softplus,   "softplus");}
KAPI  Threshold(K x) {return act(x, Cast::threshold,  "threshold");}

// -------------------------------------------------------------------------------------------
// prelu: parameterized relu
//        module accepts 1 or number of input parameters and optional initalization value
//        functional form requires weight directly rather than module's count & initial value
// -------------------------------------------------------------------------------------------
static nn::PReLUOptions prelu(K x,J i,Cast c) {
 nn::PReLUOptions o; auto m=o.num_parameters();auto w=o.init(); Pairs p; J n=xargc(x,i,p);
 if(n) TORCH_CHECK((n==1 && (xint64(x,i,m) || xdouble(x,i,w))) ||
                   (n==2 &&  xint64(x,i,m) && xdouble(x,i+1,w)),
                   "prelu: expecting 1-2 positional args in,init or (in;init)");
 while(xpair(p))
  switch(mset(p.k,c)) {
   case Setting::in:    m=int64(p,c); break;
   case Setting::init:  w=mdouble(p,c); break;
   default: AT_ERROR("prelu option: ",p.k," not recognized");
  }
 return o.num_parameters(m).init(w);
}

static K prelu(bool a,const nn::PReLUOptions& o) {
 K x=KDICT; nn::PReLUOptions d;
 if(a || d.num_parameters() != o.num_parameters()) OPTION(x, in,   kj(o.num_parameters()));
 if(a || d.init()           != o.init())           OPTION(x, init, kf(o.init()));
 return x;
}

KAPI Prelu(K x) {
 KTRY
  bool p; Tensor t,w;
  if(!x->t && x->n==2)
   p=xtenarg(x,t,w);
  else if(0<x->t && x->t<98 && x->n==2)
   p=false, t=kput(x), w=t[1], t=t[0];
  else
   AT_ERROR("prelu expects 2 args: input & weight, received ",kname(x->t),", count: ",xlen(x));
  return kresult(p, torch::prelu(t,w));
 KCATCH("prelu");
}

// ----------------------------------------------------------------------------------------------------
// distance funtions: could be considered layers or cost functions, so not declared static here
// similar - cosine similarity distance, parse/retrieve optional dimension and epsilon
// pairwise - pairwise distance, parse/retrieve optional power, eps, deep dimension flag
// ----------------------------------------------------------------------------------------------------
nn::CosineSimilarityOptions similar(K x,J i,Cast c) {
 Pairs p; J n=xargc(x,i,p); nn::CosineSimilarityOptions o;
 for(J j=0;j<n;++j)
  switch(j) {
   case 0: o.dim(int64(x,i+j,c,Setting::dim)); break;
   case 1: o.eps(mdouble(x,i+j,c,Setting::eps)); break;
   default: AT_ERROR(msym(c),": unrecognized positional arg(s), up to 2 args(dim,eps) expected, ",n," supplied");
  }
 while(xpair(p))
  switch(mset(p.k,c)) {
   case Setting::dim: o.dim(int64(p,c)); break;
   case Setting::eps: o.eps(mdouble(p,c)); break;
   default: AT_ERROR("unrecognized option: ",p.k," for cosine similarity distance");
  }
 return o;
}

void similar(bool a,K x,const nn::CosineSimilarityOptions& o) {
 nn::CosineSimilarityOptions d; 
 if(a || (o.dim() != o.dim())) OPTION(x, dim, kj(o.dim()));
 if(a || (o.eps() != d.eps())) OPTION(x, eps, kf(o.eps()));
}

static K similar(bool a,const nn::CosineSimilarityOptions& o) {K x=KDICT; similar(a,x,o); return x;}

nn::PairwiseDistanceOptions pairwise(K x,J i,Cast c) {
 Pairs p; J n=xargc(x,i,p); nn::PairwiseDistanceOptions o;
 for(J j=0;j<n;++j)
  switch(j) {
   case 0: o.p(mdouble(x,i+j,c,Setting::p)); break;
   case 1: o.eps(mdouble(x,i+j,c,Setting::eps)); break;
   case 2: o.keepdim(mbool(x,i+j,c,Setting::keepdim)); break;
   default: AT_ERROR(msym(c),": unrecognized positional arg(s), up to 3 args(p,eps,keepdim) expected, ",n," supplied");
  }
 while(xpair(p))
  switch(mset(p.k,c)) {
   case Setting::p: o.p(mdouble(p,c)); break;
   case Setting::eps: o.eps(mdouble(p,c)); break;
   case Setting::keepdim: o.keepdim(mbool(p,c)); break;
   default: AT_ERROR("unrecognized option: ",p.k," for pairwise distance");
  }
 return o;
}

void pairwise(bool a,K x,const nn::PairwiseDistanceOptions& o) {
 nn::PairwiseDistanceOptions d; 
 if(a || (o.p()       != d.p()))       OPTION(x, p,       kf(o.p()));
 if(a || (o.eps()     != d.eps()))     OPTION(x, eps,     kf(o.eps()));
 if(a || (o.keepdim() != d.keepdim())) OPTION(x, keepdim, kb(o.keepdim()));
}

static K pairwise(bool a,const nn::PairwiseDistanceOptions& o) {K x=KDICT; pairwise(a,x,o); return x;}

// ------------------------------------------------------------------------
// functional form of the distance calculations
// ------------------------------------------------------------------------
static K distance(K x,Cast c) {
 KTRY
  TORCH_CHECK(!x->t, msym(c)," not implemented for ",kname(x->t));
  Tensor r, *a=xten(x,0), *b=xten(x,1);
  switch(c) {
   case Cast::pairwise: r=fnn::pairwise_distance(a ? *a : kput(x,0), b ? *b : kput(x,1), pairwise(x,2,c)); break;
   case Cast::similar:  r=fnn::cosine_similarity(a ? *a : kput(x,0), b ? *b : kput(x,1), similar(x,2,c)); break;
   default: AT_ERROR("unrecognized distance function");
  }
  return kresult(a||b,r);
 KCATCH("distance");
}

KAPI Pairwise(K x) {return distance(x,Cast::pairwise);}
KAPI Similar(K x)  {return distance(x,Cast::similar);}

KAPI pdist(K x) {
 KTRY
  TORCH_CHECK(!x->t, "pdist not implemented for ",kname(x->t));
  F p=2; bool b=x->n==2 && xnum(x,1,p); Tensor *t = b ? xten(x,0) : xten(x);
  return kresult(t, torch::pdist(t ? *t : (b ? kput(x,0) : kput(x)), p));
 KCATCH("pdist");
}

// ----------------------------------------------------------------------------------------------------
// flatten - process arg(s) from k and return options
//         - return options used given a flatten module used
//         - call flatten as function given input/tensor and optional start & end dimensions
// ----------------------------------------------------------------------------------------------------
static nn::FlattenOptions flatten(K x,J i,Cast c) {
 nn::FlattenOptions o; int64_t s=o.start_dim(),e=o.end_dim(); Pairs p; J n=xargc(x,i,p);
 if(!(n==0 || (xint64(x,i,s) && (n==1 || (n==2 && xint64(x,i+1,e))))))
  AT_ERROR("flatten: unrecognized arg(s)");
 while(xpair(p))
  switch(mset(p.k,c)) {
   case Setting::start: s=plong(p); break;
   case Setting::end:   e=plong(p); break;
   default: AT_ERROR("flatten option: ",p.k," not recognized");
  }
 return o.start_dim(s).end_dim(e);
}

static K flatten(bool a,const nn::FlattenOptions& o) {
 K x=KDICT; nn::FlattenOptions d;
 if(a || d.start_dim() != o.start_dim()) OPTION(x, start, kj(o.start_dim()));
 if(a || d.end_dim()   != o.end_dim())   OPTION(x, end,   kj(o.end_dim()));
 return x;
}

KAPI kflatten(K x) {
 KTRY
  bool m=false; Tensor t;
  auto o=flatten((xten(x,t) || xten(x,0,t) || (m=xmixed(x,3))) ? x : nullptr, 1, Cast::flatten);
  if(t.defined())
   return kten(torch::flatten(t, o.start_dim(), o.end_dim()));
  else
   return kget(torch::flatten(m ? kput(x,0) : kput(x), o.start_dim(), o.end_dim()));
 KCATCH("flatten");
}

// ----------------------------------------------------------------------------------------------------
// squeeze/unsqueeze - squeeze works with/without a dimension specified, unsqueeze requires it
// ----------------------------------------------------------------------------------------------------
static SqueezeOptions squeeze(K x,J i,Cast c) {
 SqueezeOptions o; Pairs p; bool b; J n=xargc(x,i,p);
 if(n==1) {
  if(xbool(x,i,b))
    o.inplace(b);
  else
    o.dim(int64n(x,i,c,Setting::dim));
 } else if(n==2) {
   o.dim(   int64n(x,i,   c, Setting::dim));
   o.inplace(mbool(x,i+1, c, Setting::inplace));
 } else if(n) {
  AT_ERROR(msym(c), ": unrecognized positional arg(s), expecting dim, inplace flag, or (dim;inplace flag)");
 }
 while(xpair(p))
  switch(mset(p.k,c)) {
   case Setting::dim:     o.dim(int64n(p,c)); break;
   case Setting::inplace: o.inplace(mbool(p,c)); break;
   default: AT_ERROR(msym(c)," option: ",p.k," not recognized");
  }
 TORCH_CHECK(c==Cast::squeeze || o.dim().has_value(), msym(c),": no dimension given");
 return o;
}

static K squeeze(bool a,const SqueezeOptions& o) {
 K x=KDICT;
 if(o.dim().has_value()) OPTION(x, dim,     kj(o.dim().value()));
 if(a || o.inplace())    OPTION(x, inplace, kb(o.inplace()));
 return x;
}
 
// --------------------------------------------------------------------------------------
// attention - parse/retrieve settings for multi head attention
// --------------------------------------------------------------------------------------
static nn::MultiheadAttentionOptions attention(K x,J i,Cast c) {
 Pairs p; J n=xargc(x,i,p); nn::MultiheadAttentionOptions o(0,0);
 for(J j=0;j<n;++j) {
   switch(j) {
    case 0: o.embed_dim(int64(x,i+j,c,Setting::dim)); break;
    case 1: o.num_heads(int64(x,i+j,c,Setting::heads)); break;
    case 2: o.dropout(mdouble(x,i+j,c,Setting::dropout)); break;
    case 3: o.bias(mbool(x,i+j,c,Setting::bias)); break;
    case 4: o.add_bias_kv(mbool(x,i+j,c,Setting::addbias)); break;
    case 5: o.add_zero_attn(mbool(x,i+j,c,Setting::addzero)); break;
    case 6: o.kdim(int64(x,i+j,c,Setting::kdim)); break;
    case 7: o.vdim(int64(x,i+j,c,Setting::vdim)); break;
    default: AT_ERROR(msym(c),": up to 8 positional arguments expected, ",n," given");
  }
 }
 while(xpair(p))
  switch(mset(p.k,c)) {
   case Setting::dim:     o.embed_dim(int64(p,c)); break;
   case Setting::heads:   o.num_heads(int64(p,c)); break;
   case Setting::dropout: o.dropout(mdouble(p,c)); break;
   case Setting::bias:    o.bias(mbool(p,c)); break;
   case Setting::addbias: o.add_bias_kv(mbool(p,c)); break;
   case Setting::addzero: o.add_zero_attn(mbool(p,c)); break;
   case Setting::kdim:    o.kdim(int64(p,c)); break;
   case Setting::vdim:    o.vdim(int64(p,c)); break;
   default: AT_ERROR("unrecognized ",msym(c)," option: ",p.k); break;
  }
 TORCH_CHECK(o.embed_dim()>0, msym(c), ": positive embedding dimension required");
 TORCH_CHECK(o.num_heads()>0, msym(c), ": positive number of heads required");
 if(o.kdim()<=0) o.kdim(o.embed_dim());
 if(o.vdim()<=0) o.vdim(o.embed_dim());
 return o;
}

static K attention(bool a,const nn::MultiheadAttentionOptions& o) {
 K x=KDICT; nn::MultiheadAttentionOptions d(o.embed_dim(),o.num_heads());
 OPTION(x, dim,   kj(o.embed_dim()));
 OPTION(x, heads, kj(o.num_heads()));
 if(a || (o.dropout()       != d.dropout()))       OPTION(x, dropout, kf(o.dropout()));
 if(a || (o.bias()          != d.bias()))          OPTION(x, bias,    kb(o.bias()));
 if(a || (o.add_bias_kv()   != d.add_bias_kv()))   OPTION(x, addbias, kb(o.add_bias_kv()));
 if(a || (o.add_zero_attn() != d.add_zero_attn())) OPTION(x, addzero, kb(o.add_zero_attn()));
 if(a || (o.kdim()          != d.kdim()))          OPTION(x, kdim,    kj(o.kdim()));
 if(a || (o.vdim()          != d.vdim()))          OPTION(x, vdim,    kj(o.vdim()));
 return x;
}

// --------------------------------------------------------------------------------------
// encode/decode layers - parse/retrieve settings for multi head attention
// --------------------------------------------------------------------------------------
static c10::variant<torch::enumtype::kReLU, torch::enumtype::kGELU> codefn(Cast c,S s) {
 switch(emap(s)) {
  case Enum::relu: return torch::kReLU;
  case Enum::gelu: return torch::kGELU;
  default: AT_ERROR("unrecognized ", msym(c), " activation fn: ",s); break;
 }
}

template<typename O>static O codelayer(K x,J i,Cast c) {
 TORCH_CHECK(x->t>=0 || x->t==99, msym(c),": unrecognized  or insufficient arg(s), ",kname(x),", ",kstring(x));
 Pairs p; J n=xargc(x,i,p); O o(0,0);
 for(J j=0;j<n;++j) {
  switch(j) {
   case 0: o.d_model(int64(x,i+j,c,Setting::in)); break;
   case 1: o.nhead(int64(x,i+j,c,Setting::heads)); break;
   case 2: o.dim_feedforward(int64(x,i+j,c,Setting::dim)); break;
   case 3: o.dropout(mdouble(x,i+j,c,Setting::dropout)); break;
   case 4: o.activation(codefn(c,code(x,i+j,c,Setting::fn))); break;
   default: AT_ERROR(msym(c),": up to 5 positional arguments expected, ",n," given");
  }
 }
 while(xpair(p))
  switch(mset(p.k,c)) {
   case Setting::in:      o.d_model(int64(p,c)); break;
   case Setting::heads:   o.nhead(int64(p,c)); break;
   case Setting::dim:     o.dim_feedforward(int64(p,c)); break;
   case Setting::dropout: o.dropout(mdouble(p,c)); break;
   case Setting::fn:      o.activation(codefn(c,code(p,c))); break;
   default: AT_ERROR("unrecognized ",msym(c)," option: ",p.k); break;
  }
 TORCH_CHECK(o.d_model()>0, msym(c), ": positive number of input features required");
 TORCH_CHECK(  o.nhead()>0, msym(c), ": positive number of heads required");
 return o;
}

template<typename O>static K codelayer(bool a,const O& o) {
 K x=KDICT; O d(o.d_model(),o.nhead());
 OPTION(x, in,    kj(o.d_model()));
 OPTION(x, heads, kj(o.nhead()));
 if(a || (o.dim_feedforward()    != d.dim_feedforward()))    OPTION(x, dim,     kj(o.dim_feedforward()));
 if(a || (o.dropout()            != d.dropout()))            OPTION(x, dropout, kf(o.dropout()));
 if(a || (o.activation().index() != d.activation().index())) OPTION(x, fn,      ks(ESYM(o.activation())));
 return x;
}

// ----------------------------------------------------------------------------------------------------
// create transformer encoder/decoder layers:
// codeoff - get offset for processing sum-module arg(s), e.g. (`encodelayer;512;8) -> offset=1
// codelayer - process a created layer module or args to define one
// codenorm - process a created layernorm module or arg(s) to define one
// coder - template for creating encoder/decoder layers with submodule, layer count, optional norm 
// encoder,decoder - invoke template 'coder' with encoder/decoder layer & option types
// ----------------------------------------------------------------------------------------------------
static J codeoff(K x,Cast c) {
 J i=0; S s=nullptr;
 switch(x->t) {
  case 0:   if(xsym(x,0,s)) i=xsym(x,1) ? 2 : 1; break;
  case -KS: s=x->s, i=1; break;
  case KS:  if(x->n) s=kS(x)[0], i=(2<x->n) ? 2 : x->n; break;
  case 99:  i=-1; break;
 }
 if(s) { // module sym not required; if given, check if one of `encoderlayer`decoderlayer`layernorm
  auto m=msym(s);
  TORCH_CHECK(m==Cast::layernorm || m==(c==Cast::encoder ? Cast::encoderlayer : Cast::decoderlayer),
              msym(c), ": unexpected layer '", s, "'");
 }
 return i;
}
 
template<typename L,typename O> static L codelayer(K x,Cast c,std::vector<K>& v) {
 Kmodule *m=xmodule(x); bool e=c == Cast::encoder;
 if(m) {
  auto *l=mref(m->m).as<L>();
  TORCH_CHECK(l, msym(c),": expecting ",(e ? "encoding layer" : "decoding layer"),
                         " given ",mlabel(mref(m->m))," module");
  v.push_back(x);
  return L(*l);
 } else {
  return L(codelayer<O>(x,codeoff(x,c),e ? Cast::encoderlayer : Cast::decoderlayer));
 }
}

static nn::LayerNorm codenorm(K x,J n,Cast c,std::vector<K>& v) {
 if(x) {
  auto *m=xmodule(x);
  if(m) {
   auto *l=mref(m->m).as<nn::LayerNorm>();
   TORCH_CHECK(l, msym(c),": expecting normalization layer, given ",mlabel(mref(m->m))," module");
   v.push_back(x);
   return nn::LayerNorm(*l);
  } else {
   return nn::LayerNorm(x->t==-KJ ? nn::LayerNormOptions({x->j}) : layernorm(x,codeoff(x,c),Cast::layernorm));
  }
 } else {
  return nn::LayerNorm(nn::LayerNormOptions({n}));
 }
}

template<typename R,typename L,typename O>static R coder(K x,J i,Cast c) {
 TORCH_CHECK(x->t==0 || x->t==99, msym(c),": unrecognized  or insufficient arg(s), ",kname(x),", ",kstring(x));
 Pairs p; J l=-1,n=xargc(x,i,p); L m1=nullptr; nn::LayerNorm m2=nullptr; std::vector<K> v;
 for(J j=0;j<n;++j) {
  K y=kK(x)[i+j];
  switch(j) {
   case 0: m1=codelayer<L,O>(y,c,v); break;
   case 1: l=int64(x,i+j,c,Setting::layers); break;
   case 2: if(!xempty(y)) m2=codenorm(x,0,c,v); break;
   default: AT_ERROR(msym(c),": up to 3 positional arguments(layer args;number of layers;norm args) expected, ",n," given");
  }
 }
 while(xpair(p)) {
  switch(mset(p.k,c)) {
   case Setting::decoderlayer:
    TORCH_CHECK(c==Cast::decoder, msym(c),": cannot create a decoder layer");
    TORCH_CHECK(p.t>=0, msym(c),": unrecognized arg(s) for decoder layer (",kname(p.t),")");
    m1=codelayer<L,O>(p.v,c,v);
    break;
   case Setting::encoderlayer:
    TORCH_CHECK(c==Cast::encoder, msym(c),": cannot create an encoder layer");
    TORCH_CHECK(p.t>=0, msym(c),": unrecognized arg(s) for encoder layer (",kname(p.t),")");
    m1=codelayer<L,O>(p.v,c,v);
    break;
   case Setting::layers:
    l=int64(p,c);
    break;
   case Setting::layernorm:
    if(!pempty(p)) {
     TORCH_CHECK(p.t==-KJ || p.t>=0, msym(c),": unrecognized arg(s) for layer normalization (",kname(p.t),")");
     m2=codenorm(p.t==-KJ ? nullptr : p.v, p.j, c, v);
    }
    break;
   default: AT_ERROR("unrecognized ",msym(c)," option: ",p.k); break;
  }
 }
 TORCH_CHECK(l>=0, msym(c), ": non-negative number of layers must be defined");
 auto r=m2.is_empty() ? R(m1,l) : R(m1,l).norm(AnyModule(m2));
 if(v.size()) kfree(v);
 return r;
}

static nn::TransformerDecoderOptions decoder(K x,J i,Cast c) {
 return coder<nn::TransformerDecoderOptions,
              nn::TransformerDecoderLayer,  
              nn::TransformerDecoderLayerOptions>(x,i,c);
}  

static nn::TransformerEncoderOptions encoder(K x,J i,Cast c) {
 return coder<nn::TransformerEncoderOptions,
              nn::TransformerEncoderLayer,  
              nn::TransformerEncoderLayerOptions>(x,i,c);
}  

// -----------------------------------------------------------------------------
// codenorm - retrieve dictionary of options in layer norm module if not empty
// coder - templated retrieval of common options of encoder/decoder layer
// decoder,encoder - retrieve options of encoder/decoder layer,layer count,norm
// -----------------------------------------------------------------------------
static K codenorm(bool a,const AnyModule& m) {
 return m.is_empty() ? KDICT : layernorm(a,m.get<nn::LayerNorm>()->options);
}

template<typename O> static void coder(bool a,K x,const O& o) {
 OPTION(x, layers,    kj(o.num_layers()));
 OPTION(x, layernorm, codenorm(a,o.norm()));
}

static K decoder(bool a,const nn::TransformerDecoderOptions& o) {
 K x=KDICT; OPTION(x, decoderlayer, codelayer(a,o.decoder_layer()->options)); coder(a,x,o); return x;
}

static K encoder(bool a,const nn::TransformerEncoderOptions& o) {
 K x=KDICT; OPTION(x, encoderlayer, codelayer(a,o.encoder_layer()->options)); coder(a,x,o); return x;
}

// -------------------------------------------------------------------------------------
// customcoder - create or retrieve options from custom encoder/decoder for transformer
// transformer - create transformer or retrieve options from existing transformer module
// -------------------------------------------------------------------------------------
static AnyModule customcoder(K x,Setting t,std::vector<K>& v) {
 K y; J i; S s,nm; AnyModule a; Kmodule *m=xmodule(x);
 if(m) {
  TORCH_CHECK(c10::holds_alternative<AnyModule>(m->m), "unable to add ",mlabel(mref(m->m))," module as custom ",mset(t));
  a=c10::get<AnyModule>(m->m).clone();
  v.push_back(x);
 } else {
  if(xdict(x)) {
   i=-1; s=statemodule(x,i); nm=statename(x,i), y=stateoptions(x,i);
  } else {
   y=x; msyms(y,s,nm); i=argstart(y,nm);
  }
  a=anymodule(y,i,msym(s));
 }
 return a;
}

static nn::TransformerOptions transformer(K x,J i,Cast c) {
 Pairs p; Setting s; J n=xargc(x,i,p); nn::TransformerOptions o; std::vector<K> v;
 for(J j=0;j<n;++j) {
  switch(j) {
   case 0: o.d_model(int64(x,i+j,c,Setting::in)); break;
   case 1: o.nhead(int64(x,i+j,c,Setting::heads)); break;
   case 2: o.num_encoder_layers(int64(x,i+j,c,Setting::elayers)); break;
   case 3: o.num_decoder_layers(int64(x,i+j,c,Setting::dlayers)); break;
   case 4: o.dim_feedforward(int64(x,i+j,c,Setting::dim)); break;
   case 5: o.dropout(mdouble(x,i+j,c,Setting::dropout)); break;
   case 6: o.activation(codefn(c,code(x,i+j,c,Setting::fn))); break;
   case 7: if(!xempty(x,i+j)) o.custom_encoder(customcoder(kK(x)[i+j],Setting::encoder,v)); break;
   case 8: if(!xempty(x,i+j)) o.custom_decoder(customcoder(kK(x)[i+j],Setting::decoder,v)); break;
   default: AT_ERROR(msym(c),": up to 9 positional arguments expected, ",n," given");
  }
 }
 while(xpair(p))
  switch((s=mset(p.k,c))) {
   case Setting::in:      o.d_model(int64(p,c)); break;
   case Setting::heads:   o.nhead(int64(p,c)); break;
   case Setting::elayers: o.num_encoder_layers(int64(p,c)); break;
   case Setting::dlayers: o.num_decoder_layers(int64(p,c)); break;
   case Setting::dim:     o.dim_feedforward(int64(p,c)); break;
   case Setting::dropout: o.dropout(mdouble(p,c)); break;
   case Setting::fn:      o.activation(codefn(c,code(p,c))); break;
   case Setting::encoder: 
   case Setting::decoder:
    if(!pempty(p)) {
     TORCH_CHECK(p.t==-KS || p.t>=0, msym(c), ": unrecognized arg(s) for custom ",p.k);
     auto a=p.t==-KS ? anymodule(nullptr,0,msym(p.s)) : customcoder(p.v,s,v);
     s==Setting::encoder ?  o.custom_encoder(a) : o.custom_decoder(a);
    }
    break;
   default: AT_ERROR("unrecognized ",msym(c)," option: ",p.k); break;
  }
 TORCH_CHECK(o.d_model()>0, msym(c), ": positive number of input features required");
 TORCH_CHECK(  o.nhead()>0, msym(c), ": positive number of heads required");
 if(v.size()) kfree(v);  // free any allocated submodules
 return o;
}

static K customcoder(bool a,const AnyModule& m) {
 K x,k=ktn(KS,2),v=ktn(0,2); Cast c;
 std::tie(c,x)=mopt(a,*m.ptr());
 kS(k)[0]=statekey(State::module);  kK(v)[0]=ks(msym(c));
 kS(k)[1]=statekey(State::options); kK(v)[1]=x;
 return xD(k,v);
}

static K transformer(bool a,const nn::TransformerOptions& o) {
 K x=KDICT; nn::TransformerOptions d;
 if(a || (o.d_model()            != d.d_model()))            OPTION(x, in,      kj(o.d_model()));
 if(a || (o.nhead()              != d.nhead()))              OPTION(x, heads,   kj(o.nhead()));
 if(a || (o.num_encoder_layers() != d.num_encoder_layers())) OPTION(x, elayers, kj(o.num_encoder_layers()));
 if(a || (o.num_decoder_layers() != d.num_decoder_layers())) OPTION(x, dlayers, kj(o.num_decoder_layers()));
 if(a || (o.dim_feedforward()    != d.dim_feedforward()))    OPTION(x, dim,     kj(o.dim_feedforward()));
 if(a || (o.dropout()            != d.dropout()))            OPTION(x, dropout, kf(o.dropout()));
 if(a || (o.activation().index() != d.activation().index())) OPTION(x, fn,      ks(ESYM(o.activation())));
 if(o.custom_encoder().is_empty()) {
  if(a) OPTION(x, encoder, KDICT);
 } else { 
  OPTION(x, encoder, customcoder(a,o.custom_encoder()));
 }
 if(o.custom_decoder().is_empty()) {
  if(a) OPTION(x, decoder, KDICT);
 } else { 
  OPTION(x, decoder, customcoder(a,o.custom_decoder()));
 }
 return x;
}

// ----------------------------------------------------------------------------------------------------
// getsize - get size(s) for expand & reshape
// expand
// reshape
// ----------------------------------------------------------------------------------------------------
static SizeOptions getsize(K x,J i,Cast c) {
 SizeOptions o({}); Pairs p; J n=xargc(x,i,p);
 TORCH_CHECK(n<2, msym(c),": 1 positional argument expected, ",n," given");
 if(n==1) o.size(mlongs(x,i,c,Setting::size));
 while(xpair(p))
  switch(mset(p.k,c)) {
   case Setting::size: o.size(mlongs(p,c)); break;
   default: AT_ERROR(msym(c)," option: ",p.k," not recognized");
  }
 return o;
}

static K getsize(bool a,const SizeOptions& o) {
 K x=KDICT;
 OPTION(x, size, klist(o.size().size(),o.size().data()));
 return x;
}

// ----------------------------------------------------------------------------------------------------
// anymodule - define module from supplied options, return as generic AnyModule 
// ----------------------------------------------------------------------------------------------------
static AnyModule anymodule(K x,J i,Cast c) {
 switch(c) {
  case Cast::batchnorm1d:  return AnyModule(nn::BatchNorm1d(batchnorm<nn::BatchNormOptions>(x,i,c)));
  case Cast::batchnorm2d:  return AnyModule(nn::BatchNorm2d(batchnorm<nn::BatchNormOptions>(x,i,c)));
  case Cast::batchnorm3d:  return AnyModule(nn::BatchNorm3d(batchnorm<nn::BatchNormOptions>(x,i,c)));

  case Cast::instancenorm1d:  return AnyModule(nn::InstanceNorm1d(batchnorm<nn::InstanceNormOptions>(x,i,c)));
  case Cast::instancenorm2d:  return AnyModule(nn::InstanceNorm2d(batchnorm<nn::InstanceNormOptions>(x,i,c)));
  case Cast::instancenorm3d:  return AnyModule(nn::InstanceNorm3d(batchnorm<nn::InstanceNormOptions>(x,i,c)));

  case Cast::groupnorm:  return AnyModule(nn::GroupNorm(groupnorm(x,i,c)));
  case Cast::layernorm:  return AnyModule(nn::LayerNorm(layernorm(x,i,c)));
  case Cast::localnorm:  return AnyModule(nn::LocalResponseNorm(localnorm<nn::LocalResponseNormOptions>(x,i,c)));
  case Cast::crossmap2d: return AnyModule(nn::CrossMapLRN2d(localnorm<nn::CrossMapLRN2dOptions>(x,i,c)));

  case Cast::embed:        return AnyModule(embed(x,i,c));
  case Cast::embedbag:     return AnyModule(embedbag(x,i,c));
  case Cast::linear:       return AnyModule(nn::Linear(linear(x,i,c)));
  case Cast::bilinear:     return AnyModule(nn::Bilinear(bilinear(x,i,c)));

  case Cast::drop:         return AnyModule(nn::Dropout(drop(x,i,c)));
  case Cast::drop2d:       return AnyModule(nn::Dropout2d(drop(x,i,c)));
  case Cast::drop3d:       return AnyModule(nn::Dropout3d(drop(x,i,c)));
  case Cast::adrop:        return AnyModule(nn::AlphaDropout(drop(x,i,c)));
  case Cast::fadrop:       return AnyModule(nn::FeatureAlphaDropout(drop(x,i,c)));

  case Cast::conv1d:       return AnyModule(nn::Conv1d(conv<1>(x,i,c)));
  case Cast::conv2d:       return AnyModule(nn::Conv2d(conv<2>(x,i,c)));
  case Cast::conv3d:       return AnyModule(nn::Conv3d(conv<3>(x,i,c)));

  case Cast::convtranspose1d:  return AnyModule(nn::ConvTranspose1d(convtran<1>(x,i,c)));
  case Cast::convtranspose2d:  return AnyModule(nn::ConvTranspose2d(convtran<2>(x,i,c)));
  case Cast::convtranspose3d:  return AnyModule(nn::ConvTranspose3d(convtran<3>(x,i,c)));

  case Cast::fold:         return AnyModule(nn::Fold(fold(x,i,c)));
  case Cast::unfold:       return AnyModule(nn::Unfold(unfold(x,i,c)));
  case Cast::upsample:     return AnyModule(nn::Upsample(upsample<nn::UpsampleOptions>(x,i,c)));

  case Cast::maxpool1d:    return AnyModule(nn::MaxPool1d(maxpool<1>(x,i,c)));
  case Cast::maxpool2d:    return AnyModule(nn::MaxPool2d(maxpool<2>(x,i,c)));
  case Cast::maxpool3d:    return AnyModule(nn::MaxPool3d(maxpool<3>(x,i,c)));

  case Cast::avgpool1d:    return AnyModule(nn::AvgPool1d(avgpool<1>(x,i,c)));
  case Cast::avgpool2d:    return AnyModule(nn::AvgPool2d(avgpool<2>(x,i,c)));
  case Cast::avgpool3d:    return AnyModule(nn::AvgPool3d(avgpool<3>(x,i,c)));

  case Cast::adaptmax1d:   return AnyModule(nn::AdaptiveMaxPool1d(adapt<1,nn::AdaptiveMaxPool1dOptions>(x,i,c)));
  case Cast::adaptmax2d:   return AnyModule(nn::AdaptiveMaxPool2d(adapt<2,nn::AdaptiveMaxPool2dOptions>(x,i,c)));
  case Cast::adaptmax3d:   return AnyModule(nn::AdaptiveMaxPool3d(adapt<3,nn::AdaptiveMaxPool3dOptions>(x,i,c)));

  case Cast::adaptavg1d:   return AnyModule(nn::AdaptiveAvgPool1d(adapt<1,nn::AdaptiveAvgPool1dOptions>(x,i,c)));
  case Cast::adaptavg2d:   return AnyModule(nn::AdaptiveAvgPool2d(adapt<2,nn::AdaptiveAvgPool2dOptions>(x,i,c)));
  case Cast::adaptavg3d:   return AnyModule(nn::AdaptiveAvgPool3d(adapt<3,nn::AdaptiveAvgPool3dOptions>(x,i,c)));

  case Cast::fmaxpool2d:   return AnyModule(nn::FractionalMaxPool2d(fpool<2>(x,i,c)));
  case Cast::fmaxpool3d:   return AnyModule(nn::FractionalMaxPool3d(fpool<3>(x,i,c)));

  case Cast::lppool1d:     return AnyModule(nn::LPPool1d(lppool<1>(x,i,c)));
  case Cast::lppool2d:     return AnyModule(nn::LPPool2d(lppool<2>(x,i,c)));

  case Cast::pad:          return AnyModule(Pad(pad(x,i,c)));
  case Cast::pad1d:        return AnyModule(nn::ConstantPad1d(cpad<1,nn::ConstantPad1dOptions>(x,i,c)));
  case Cast::pad2d:        return AnyModule(nn::ConstantPad2d(cpad<2,nn::ConstantPad2dOptions>(x,i,c)));
  case Cast::pad3d:        return AnyModule(nn::ConstantPad3d(cpad<3,nn::ConstantPad3dOptions>(x,i,c)));
  case Cast::reflect1d:    return AnyModule(nn::ReflectionPad1d(npad<1,nn::ReflectionPad1dOptions>(x,i,c)));
  case Cast::reflect2d:    return AnyModule(nn::ReflectionPad2d(npad<2,nn::ReflectionPad2dOptions>(x,i,c)));
  case Cast::replicate1d:  return AnyModule(nn::ReplicationPad1d(npad<1,nn::ReplicationPad1dOptions>(x,i,c)));
  case Cast::replicate2d:  return AnyModule(nn::ReplicationPad2d(npad<2,nn::ReplicationPad2dOptions>(x,i,c)));
  case Cast::replicate3d:  return AnyModule(nn::ReplicationPad3d(npad<3,nn::ReplicationPad3dOptions>(x,i,c)));
  case Cast::zeropad2d:    return AnyModule(nn::ZeroPad2d(npad<2,nn::ZeroPad2dOptions>(x,i,c)));

  case Cast::attention:    return AnyModule(nn::MultiheadAttention(attention(x,i,c)));
  case Cast::decoderlayer: return AnyModule(nn::TransformerDecoderLayer(codelayer<nn::TransformerDecoderLayerOptions>(x,i,c)));
  case Cast::encoderlayer: return AnyModule(nn::TransformerEncoderLayer(codelayer<nn::TransformerEncoderLayerOptions>(x,i,c)));
  case Cast::decoder:      return AnyModule(nn::TransformerDecoder(decoder(x,i,c)));
  case Cast::encoder:      return AnyModule(nn::TransformerEncoder(encoder(x,i,c)));
  case Cast::transformer:  return AnyModule(nn::Transformer(transformer(x,i,c)));

  case Cast::rnn:          return AnyModule(nn::RNN(rnn(x,i,c)));
  case Cast::gru:          return AnyModule(nn::GRU(rnn<nn::GRUOptions>(x,i,c)));
  case Cast::lstm:         return AnyModule(nn::LSTM(rnn<nn::LSTMOptions>(x,i,c)));

  case Cast::identity:     noarg(c,x,i); return AnyModule(nn::Identity());
  case Cast::logsigmoid:   noarg(c,x,i); return AnyModule(nn::LogSigmoid());
  case Cast::sigmoid:      noarg(c,x,i); return AnyModule(nn::Sigmoid());
  case Cast::softsign:     noarg(c,x,i); return AnyModule(nn::Softsign());
  case Cast::softmax2d:    noarg(c,x,i); return AnyModule(nn::Softmax2d());
  case Cast::tanh:         noarg(c,x,i); return AnyModule(nn::Tanh());
  case Cast::tanhshrink:   noarg(c,x,i); return AnyModule(nn::Tanhshrink());
  case Cast::gelu:         noarg(c,x,i); return AnyModule(nn::GELU());
  case Cast::mul:          noarg(c,x,i); return AnyModule(Mul());

  case Cast::relu:         return AnyModule( nn::ReLU(inplace(x,i,c)));
  case Cast::relu6:        return AnyModule(nn::ReLU6(inplace(x,i,c)));
  case Cast::selu:         return AnyModule( nn::SELU(inplace(x,i,c)));

  case Cast::softmax:      return AnyModule(nn::Softmax(dim(x,i,c)));
  case Cast::softmin:      return AnyModule(nn::Softmin(dim(x,i,c)));
  case Cast::logsoftmax:   return AnyModule(nn::LogSoftmax(dim(x,i,c)));
  case Cast::flatten:      return AnyModule(nn::Flatten(flatten(x,i,c)));

  case Cast::squeeze:      return AnyModule(Squeeze(squeeze(x,i,c)));
  case Cast::unsqueeze:    return AnyModule(Unsqueeze(squeeze(x,i,c)));
  case Cast::expand:       return AnyModule(Expand(getsize(x,i,c)));
  case Cast::reshape:      return AnyModule(Reshape(getsize(x,i,c)));
  case Cast::cat:          return AnyModule(Cat(dim(x,i,c)));

  case Cast::elu:          return AnyModule(nn::ELU (alpha<nn::ELUOptions> (x,i,c)));
  case Cast::celu:         return AnyModule(nn::CELU(alpha<nn::CELUOptions>(x,i,c)));
  case Cast::leakyrelu:    return AnyModule(nn::LeakyReLU(slope(x,i,c)));
  case Cast::glu:          return AnyModule(nn::GLU(dim(x,i,c)));
  case Cast::hardshrink:   return AnyModule(nn::Hardshrink(lambda(x,i,c)));
  case Cast::softshrink:   return AnyModule(nn::Softshrink(lambda(x,i,c)));
  case Cast::prelu:        return AnyModule(nn::PReLU(prelu(x,i,c)));
  case Cast::rrelu:        return AnyModule(nn::RReLU(rrelu(x,i,c)));
  case Cast::hardtanh:     return AnyModule(nn::Hardtanh(hardtanh(x,i,c)));
  case Cast::softplus:     return AnyModule(nn::Softplus(softplus(x,i,c)));
  case Cast::threshold:    return AnyModule(nn::Threshold(threshold(x,i,c)));
  case Cast::pairwise:     return AnyModule(nn::PairwiseDistance(pairwise(x,i,c)));
  case Cast::similar:      return AnyModule(nn::CosineSimilarity(similar(x,i,c)));

  default:
   if(container(c))
    AT_ERROR("cannot create container module: ",msym(c));
   else
    AT_ERROR("unrecognized module: cannot create module from unrecognized enumeration ",(I)c);
 }
}

// --------------------------------------------------------------------------------------------
// mopt - given module, cast at runtime to known type and extract options as k dictionary
// --------------------------------------------------------------------------------------------
static std::tuple<Cast,K> mopt(bool a,const Module& g) { //a:all options returned if true, else only non-default
 Cast c=Cast::undefined; K x=nullptr;
 if       (g.as<Sequential>())  { c=Cast::sequential;
 } else if(g.as<SeqNest>())     { c=Cast::seqnest;
 } else if(g.as<SeqJoin>())     { c=Cast::seqjoin;
 } else if(g.as<ModuleList>())  { c=Cast::modulelist;

 } else if(auto* m=g.as<nn::BatchNorm1d>())       { c=Cast::batchnorm1d;    x=batchnorm(a,m->options);
 } else if(auto* m=g.as<nn::BatchNorm2d>())       { c=Cast::batchnorm2d;    x=batchnorm(a,m->options);
 } else if(auto* m=g.as<nn::BatchNorm3d>())       { c=Cast::batchnorm3d;    x=batchnorm(a,m->options);
 } else if(auto* m=g.as<nn::InstanceNorm1d>())    { c=Cast::instancenorm1d; x=batchnorm(a,m->options);
 } else if(auto* m=g.as<nn::InstanceNorm2d>())    { c=Cast::instancenorm2d; x=batchnorm(a,m->options);
 } else if(auto* m=g.as<nn::InstanceNorm3d>())    { c=Cast::instancenorm3d; x=batchnorm(a,m->options);
 } else if(auto* m=g.as<nn::GroupNorm>())         { c=Cast::groupnorm;      x=groupnorm(a,m->options);
 } else if(auto* m=g.as<nn::LayerNorm>())         { c=Cast::layernorm;      x=layernorm(a,m->options);
 } else if(auto* m=g.as<nn::LocalResponseNorm>()) { c=Cast::localnorm;      x=localnorm(a,c,m->options);
 } else if(auto* m=g.as<nn::CrossMapLRN2d>())     { c=Cast::crossmap2d;     x=localnorm(a,c,m->options);

 } else if(auto* m=g.as<nn::Embedding>())         { c=Cast::embed;    x=embed(a,c,m->options,m->weight);
 } else if(auto* m=g.as<nn::EmbeddingBag>())      { c=Cast::embedbag; x=embed(a,c,m->options,m->weight);
 } else if(auto* m=g.as<nn::Linear>())            { c=Cast::linear;   x=linear(a,m->options);
 } else if(auto* m=g.as<nn::Bilinear>())          { c=Cast::bilinear; x=bilinear(a,m->options);

 } else if(auto* m=g.as<nn::Dropout>())             { c=Cast::drop;   x=drop(a,m->options);
 } else if(auto* m=g.as<nn::Dropout2d>())           { c=Cast::drop2d; x=drop(a,m->options);
 } else if(auto* m=g.as<nn::Dropout3d>())           { c=Cast::drop3d; x=drop(a,m->options);
 } else if(auto* m=g.as<nn::AlphaDropout>())        { c=Cast::adrop;  x=drop(a,m->options);
 } else if(auto* m=g.as<nn::FeatureAlphaDropout>()) { c=Cast::fadrop; x=drop(a,m->options);

 } else if(auto* m=g.as<nn::Conv1d>())         { c=Cast::conv1d;          x=conv(a,m->options);
 } else if(auto* m=g.as<nn::Conv2d>())         { c=Cast::conv2d;          x=conv(a,m->options);
 } else if(auto* m=g.as<nn::Conv3d>())         { c=Cast::conv3d;          x=conv(a,m->options);
 } else if(auto* m=g.as<nn::ConvTranspose1d>()){ c=Cast::convtranspose1d; x=conv(a,m->options);
 } else if(auto* m=g.as<nn::ConvTranspose2d>()){ c=Cast::convtranspose2d; x=conv(a,m->options);
 } else if(auto* m=g.as<nn::ConvTranspose3d>()){ c=Cast::convtranspose3d; x=conv(a,m->options);

 } else if(auto* m=g.as<nn::Fold>())           { c=Cast::fold;     x=fold(a,m->options);
 } else if(auto* m=g.as<nn::Unfold>())         { c=Cast::unfold;   x=unfold(a,m->options);
 } else if(auto* m=g.as<nn::Upsample>())       { c=Cast::upsample; x=upsample(a,m->options);

 } else if(auto* m=g.as<nn::MaxPool1d>())      { c=Cast::maxpool1d; x=maxpool(a,m->options);
 } else if(auto* m=g.as<nn::MaxPool2d>())      { c=Cast::maxpool2d; x=maxpool(a,m->options);
 } else if(auto* m=g.as<nn::MaxPool3d>())      { c=Cast::maxpool3d; x=maxpool(a,m->options);

 } else if(auto* m=g.as<nn::AvgPool1d>())      { c=Cast::avgpool1d; x=avgpool(a,m->options);
 } else if(auto* m=g.as<nn::AvgPool2d>())      { c=Cast::avgpool2d; x=avgpool(a,m->options);
 } else if(auto* m=g.as<nn::AvgPool3d>())      { c=Cast::avgpool3d; x=avgpool(a,m->options);

 } else if(auto* m=g.as<nn::AdaptiveMaxPool1d>())   { c=Cast::adaptmax1d; x=adapt(m->options);
 } else if(auto* m=g.as<nn::AdaptiveMaxPool2d>())   { c=Cast::adaptmax2d; x=adapt(m->options);
 } else if(auto* m=g.as<nn::AdaptiveMaxPool3d>())   { c=Cast::adaptmax3d; x=adapt(m->options);

 } else if(auto* m=g.as<nn::AdaptiveAvgPool1d>())   { c=Cast::adaptavg1d; x=adapt(m->options);
 } else if(auto* m=g.as<nn::AdaptiveAvgPool2d>())   { c=Cast::adaptavg2d; x=adapt(m->options);
 } else if(auto* m=g.as<nn::AdaptiveAvgPool3d>())   { c=Cast::adaptavg3d; x=adapt(m->options);

 } else if(auto* m=g.as<nn::FractionalMaxPool2d>()) { c=Cast::fmaxpool2d; x=fpool(a,m->options);
 } else if(auto* m=g.as<nn::FractionalMaxPool3d>()) { c=Cast::fmaxpool3d; x=fpool(a,m->options);

 } else if(auto* m=g.as<nn::LPPool1d>())         { c=Cast::lppool1d; x=lppool(a,m->options);
 } else if(auto* m=g.as<nn::LPPool2d>())         { c=Cast::lppool2d; x=lppool(a,m->options);

 } else if(auto* m=g.as<Pad>())                  { c=Cast::pad;         x=pad(a,m);
 } else if(auto* m=g.as<nn::ConstantPad1d>())    { c=Cast::pad1d;       x=cpad(m->options);
 } else if(auto* m=g.as<nn::ConstantPad2d>())    { c=Cast::pad2d;       x=cpad(m->options);
 } else if(auto* m=g.as<nn::ConstantPad3d>())    { c=Cast::pad3d;       x=cpad(m->options);
 } else if(auto* m=g.as<nn::ReflectionPad1d>())  { c=Cast::reflect1d;   x=npad(m);
 } else if(auto* m=g.as<nn::ReflectionPad2d>())  { c=Cast::reflect2d;   x=npad(m);
 } else if(auto* m=g.as<nn::ReplicationPad1d>()) { c=Cast::replicate1d; x=npad(m);
 } else if(auto* m=g.as<nn::ReplicationPad2d>()) { c=Cast::replicate2d; x=npad(m);
 } else if(auto* m=g.as<nn::ReplicationPad3d>()) { c=Cast::replicate3d; x=npad(m);
 } else if(auto* m=g.as<nn::ZeroPad2d>())        { c=Cast::zeropad2d;   x=npad(m);

 } else if(auto* m=g.as<nn::MultiheadAttention>())      { c=Cast::attention;    x=attention(a,m->options);
 } else if(auto* m=g.as<nn::TransformerEncoderLayer>()) { c=Cast::encoderlayer; x=codelayer(a,m->options);
 } else if(auto* m=g.as<nn::TransformerDecoderLayer>()) { c=Cast::decoderlayer; x=codelayer(a,m->options);
 } else if(auto* m=g.as<nn::TransformerEncoder>())      { c=Cast::encoder;      x=encoder(a,m->options);
 } else if(auto* m=g.as<nn::TransformerDecoder>())      { c=Cast::decoder;      x=decoder(a,m->options);
 } else if(auto* m=g.as<nn::Transformer>())             { c=Cast::transformer;  x=transformer(a,m->options);

 } else if(auto* m=g.as<nn::RNN>())   { c=Cast::rnn;  x=rnn(a,m->options);
 } else if(auto* m=g.as<nn::GRU>())   { c=Cast::gru;  x=rnn(a,m->options);
 } else if(auto* m=g.as<nn::LSTM>())  { c=Cast::lstm; x=rnn(a,m->options);

 } else if(g.as<nn::Identity>())      { c=Cast::identity;
 } else if(g.as<nn::LogSigmoid>())    { c=Cast::logsigmoid;
 } else if(g.as<nn::Sigmoid>())       { c=Cast::sigmoid;
 } else if(g.as<nn::Softsign>())      { c=Cast::softsign;
 } else if(g.as<nn::Softmax2d>())     { c=Cast::softmax2d;
 } else if(g.as<nn::Tanh>())          { c=Cast::tanh;
 } else if(g.as<nn::Tanhshrink>())    { c=Cast::tanhshrink;
 } else if(g.as<nn::GELU>())          { c=Cast::gelu;
 } else if(g.as<Mul>())               { c=Cast::mul;

 } else if(auto* m=g.as<nn::ReLU>())  { c=Cast::relu;  x=inplace(a,m->options.inplace());
 } else if(auto* m=g.as<nn::SELU>())  { c=Cast::selu;  x=inplace(a,m->options.inplace());
 } else if(auto* m=g.as<nn::ReLU6>()) { c=Cast::relu6; x=inplace(a,m->options.inplace());

 } else if(auto* m=g.as<nn::Softmax>())    { c=Cast::softmax;    x=dim(a,c,m->options.dim());
 } else if(auto* m=g.as<nn::Softmin>())    { c=Cast::softmin;    x=dim(a,c,m->options.dim());
 } else if(auto* m=g.as<nn::LogSoftmax>()) { c=Cast::logsoftmax; x=dim(a,c,m->options.dim());
 } else if(auto* m=g.as<nn::Flatten>())    { c=Cast::flatten;    x=flatten(a,m->options);

 } else if(auto* m=g.as<Squeeze>())    { c=Cast::squeeze;    x=squeeze(a,m->options);
 } else if(auto* m=g.as<Unsqueeze>())  { c=Cast::unsqueeze;  x=squeeze(a,m->options);
 } else if(auto* m=g.as<Expand>())     { c=Cast::expand;     x=getsize(a,m->options);
 } else if(auto* m=g.as<Reshape>())    { c=Cast::reshape;    x=getsize(a,m->options);
 } else if(auto* m=g.as<Cat>())        { c=Cast::cat;        x=dim(a,c,m->options.dim());

 } else if(auto* m=g.as<nn::ELU>())        { c=Cast::elu;        x=alpha(a,m->options);
 } else if(auto* m=g.as<nn::CELU>())       { c=Cast::celu;       x=alpha(a,m->options);
 } else if(auto* m=g.as<nn::LeakyReLU>())  { c=Cast::leakyrelu;  x=slope(a,c,m->options);
 } else if(auto* m=g.as<nn::GLU>())        { c=Cast::glu;        x=dim(a,c,m->options.dim());
 } else if(auto* m=g.as<nn::Hardshrink>()) { c=Cast::hardshrink; x=lambda(a,c,m->options.lambda());
 } else if(auto* m=g.as<nn::Softshrink>()) { c=Cast::softshrink; x=lambda(a,c,m->options.lambda());

 } else if(auto* m=g.as<nn::PReLU>())      { c=Cast::prelu;      x=prelu(a,m->options);
 } else if(auto* m=g.as<nn::RReLU>())      { c=Cast::rrelu;      x=rrelu(a,m->options);
 } else if(auto* m=g.as<nn::Hardtanh>())   { c=Cast::hardtanh;   x=hardtanh(a,m->options);
 } else if(auto* m=g.as<nn::Softplus>())   { c=Cast::softplus;   x=softplus(a,m->options);
 } else if(auto* m=g.as<nn::Threshold>())  { c=Cast::threshold;  x=threshold(a,m->options);

 } else if(auto* m=g.as<nn::PairwiseDistance>())  { c=Cast::pairwise; x=pairwise(a,m->options);
 } else if(auto* m=g.as<nn::CosineSimilarity>())  { c=Cast::similar;  x=similar(a,m->options);
 } else if(g.as<nn::Module>()) { AT_ERROR("generic module, unable to retrieve options");
 } else { AT_ERROR("unrecognized module: ",g.name());
 }
 return std::make_tuple(c,x ? x : KDICT);
}

// ----------------------------------------------------------------------------------
// mparms - set module parms/buffers from k values in dictionary with matching names
// ----------------------------------------------------------------------------------
static void mparms(Cast c,S s,Module &m,K x,bool p) { // set named parms/buffers in module m from dict x, p true if parms
 K k=kK(x)[0],v=kK(x)[1]; Tensor V; if(v->t) V=kput(v);
 for(auto &a:p ? m.named_parameters(false) : m.named_buffers(false)) {
  J i=kfind(k,a.key());
  TORCH_CHECK(i>-1, msym(c), ": unable to find ",(p ? " parameter" : " buffer"),": ",a.key());
  Tensor t=v->t ? V[i] : kput(kK(v)[i]);
  if(a.value().defined()) {
   torch::NoGradGuard g;
   TORCH_CHECK(a.value().dtype() == t.dtype(), (s ? s : msym(c)), ": type mismatch, ", a.key(), " is ", a.value().dtype(), ", input is ", t.dtype());
   TORCH_CHECK(a.value().is_same_size(t),      (s ? s : msym(c)), ": size mismatch, ", a.key(), " is ", a.value().sizes(), ", input is ", t.sizes());
   if (a.value().device() != t.device())
    a.value().set_data(t);
   else
    a.value().set_(t);
  } else {
   a.value()=std::move(t);
  }
 }
}

static void mparms(Cast c,Module &m,K p,K f,S s=nullptr);  // s is full module name (see mfind)
static void mparms(Cast c,Module &m,K p,K f,S s) {
 if(p) mparms(c,s,m,p,true);   // if parms dictionary, set module parms from k dictionary
 if(f) mparms(c,s,m,f,false);  // if buffers defined,  set buffers from k dictionary
}

// -----------------------------------------------------------------------------------------
// addmodule - given parent & layer variants, add allowable combinations, else error
// addparent - create container, add to any previous parent layer, push on stack
// addchild - add a child layer to existing parent or push single layer to stack
// -----------------------------------------------------------------------------------------
static void addmodule(Layer& x,const Layer& y) {
 const char* s=mname(y);
 c10::visit(
  make_overload(
   [&s](Sequential& x, const SeqJoin&    y)  {if(s) x->push_back(s,y); else x->push_back(y);},
   [&s](Sequential& x, const SeqNest&    y)  {if(s) x->push_back(s,y); else x->push_back(y);},
   [&s](Sequential& x, const AnyModule&  y)  {if(s) x->push_back(s,y); else x->push_back(y);},
   [&s](SeqJoin&    x, const Sequential& y)  {if(s) x->push_back(s,y); else x->push_back(y);},
   [&s](SeqJoin&    x, const AnyModule&  y)  {if(s) x->push_back(s,y); else x->push_back(y);},
   [&s](SeqNest&    x, const SeqJoin&    y)  {if(s) x->push_back(s,y); else x->push_back(y);},
   [&s](SeqNest&    x, const SeqNest&    y)  {if(s) x->push_back(s,y); else x->push_back(y);},
   [&s](SeqNest&    x, const AnyModule&  y)  {if(s) x->push_back(s,y); else x->push_back(y);},
   []  (ModuleList& x, const auto&       y)  {x->push_back(y.ptr());},
   [](auto& x,const auto& y) {AT_ERROR("unable to add a ", mlabel(mref(y)),
                                       " module as a child of a ", mlabel(mref(x)), " module");}),
   x,y);
}

static void addname(Module& a,S s) {if(s) mname_(a)=s; else mname_(a)=c10::nullopt;}
 
static void addparent(const Layer& a,Layers& q) {
 if(q.size()) addmodule(q.top(),a);  // add to previous parent, if any
 q.push(a);                          // add new parent container to stack
}

static void addparent(Cast c,S s,Layers& q,K x=nullptr,K y=nullptr,K z=nullptr);
static void addparent(Cast c,S s,Layers& q,K x,K y,K z) {
 TORCH_CHECK(!x || xnone(x,xdict(x) ? 0 : (s ? 2 : 1)), msym(c), ": no options expected, given ", kstring(x));
 TORCH_CHECK(!(y && xlen(y)), msym(c), ": no parameters expected");
 TORCH_CHECK(!(z && xlen(z)), msym(c), ": no buffers expected");
 auto a=newcontainer(c);   // create new container module, e.g. sequential
 addname(mref(a),s);       // add name if supplied
 addparent(a,q);           // add to any previous parent, push on stack
}

static void addchild(const Layer& a,Layers& q) {
 if(q.size())
  addmodule(q.top(),a);
 else
  q.push(a);
}

static auto addchild(Cast c,S s,Layers& q,K x,K y=nullptr,K z=nullptr);
static auto addchild(Cast c,S s,Layers& q,K x,K y,K z) {
 auto a=anymodule(x,argstart(x,s),c); // create module from cast, options & offset
 auto& m=*a.ptr();                    // generic module reference
 addname(m,s);                        // add name if supplied
 if(y||z) mparms(c,m,y,z);            // add any supplied parms or buffers
 addchild(a,q);                       // add to immediate parent container on stack
 return m.modules(false).size();      // return count of all sub-modules created
}

// -------------------------------------------------------------------------------
// msuffix - compare submodule name from newly created module with stored suffix
// mcompare - compare options from two modules, return true if all match exactly
// mfind - match previous state of implicitly defined submodules 
//       - e.g. the self attention layer of an explicitly defined decoder layer
// -------------------------------------------------------------------------------
static bool msuffix(const std::string& x,const std::string& y) {
 return x.size()>=y.size() && !x.compare(x.size()-y.size(),y.size(),y);
}

static bool mcompare(Cast c,const Module& m1,const Module& m2) {
 bool b=false; Cast v,w; K x,y,z;
 std::tie(v,x)=mopt(true,m1);
 std::tie(w,y)=mopt(true,m2);
 if(v==w) {
  z=k(0,(S)"~",x,y,0); b=z->g; r0(z);
 }
 return b;
}

static void mfind(Cast c,J j,J d,S s,Layers& q,K x,K y,K z) {
 TORCH_CHECK(s, "attempting to find ",msym(c)," layer, but no name given");
 TORCH_CHECK(q.size(), "attempting to find ",msym(c)," layer at depth ",d," but no previous layer found");
 J i=0; auto& m=mref(q.top()); bool b=container(m); auto mc=m.named_children().back();
 std::string p; if(b) p=mc.key();
 for(auto& a:b ? mc.value()->named_modules(p,false) : m.named_modules(p,false)) {
  if(i==j) {
   TORCH_CHECK(msuffix(a.key(),s),"child module mismatch: ",a.key()," does not end with expected suffix '",s,"'");
   auto& m=*a.value();
   TORCH_CHECK(mcompare(c,m,container(c) ? mref(newcontainer(c)) : *anymodule(x,argstart(x,s),c).ptr()),
               "child module ", a.key(), " mismatch with given options");
   if(y||z) mparms(c,m,y,z,(S)a.key().c_str());   // reset any supplied parms or buffers
   return;
  }
  i++;
 }
 AT_ERROR("unable to find ",msym(c)," layer named ",s," in parent ",mlabel(b ? *mc.value() : m)," layer at depth ",d);
}

// --------------------------------------------------------------------------------------------
// mdepth - check given depth, must be non-zero if stack populated, no greater than stack size
// mpush - add new parent/child module to network stored in stack of layers
// --------------------------------------------------------------------------------------------
static void mdepth(Cast c,size_t d,Layers& q) {
 TORCH_CHECK(d >=(q.size() ? 1 : 0), msym(c), ": depth ",d," below min depth of ",q.size() ? 1 : 0);
 TORCH_CHECK(d <= q.size(),          msym(c), ": depth ",d," above max depth of ",q.size());
 while(q.size()>d) q.pop();
}

static std::tuple<Cast,J> mpush(Layers& q,J j,J d,S s,S nm,K x,K y=nullptr,K z=nullptr);
static std::tuple<Cast,J> mpush(Layers& q,J j,J d,S s,S nm,K x,K y,K z) {
 Cast c=msym(s); J n=q.size();
 if(d>n || (n && !container(q.top()))) {
  mfind(c,j,d,nm,q,x,y,z); j++;  // previous module has self-contained child modules
 } else {
  mdepth(c,d,q); j=0;
  if(container(c))
   addparent(c,nm,q,x,y,z);
  else
   addchild(c,nm,q,x,y,z);
 }
 return std::make_tuple(c,j);
}

static Cast mpush(Layers& q,J d,K x) {
 J j; S s,nm; Cast c;
 msyms(x,s,nm);
 std::tie(c,j)=mpush(q,0,d,s,nm,x);
 return c;}

// -------------------------------------------------------------------------------
// mtree - parse nested tree of layers -- type,name,options -- to build modules
// mdv - parse (depth;value) pair(s) to build module(s)
// mtable - module(s) from table of options & depth, optional name,parms & buffers
// mextend - add a created module to existing module(s) at optional depth
// -------------------------------------------------------------------------------
static Cast mtree(K x,size_t d,Layers& q) {
 K y=x->t || !x->n ? x : kK(x)[0];
 Cast c=mpush(q,d,y);    // get type of overall container module
 if(!x->t)               // process any child modules
  for(J i=1;i<x->n;i++)
   mtree(kK(x)[i],d+1,q);
 return c;
}

static K mtree(K x,J d=0,Kmodule *l=nullptr); // higher-level call, can add to existing module
static K mtree(K x,J d,Kmodule *l) {
 Layers q; if(l) mstack(l,q);
 Cast c=mtree(x,d ? d : q.size(),q);
 return l ? (K)0 : rootlayer(c,q);
}

static Cast mdv(K x,J n,Layers& q) { // process n depth-value pairs, n=-1 if one, e.g. (1;(`linear;784;10))
 Cast c,p=Cast::undefined; J d,m=n<0 ? 0 : n; K v;
 for(J i=n<0 ? -1 : 0;i<m;++i) {
  d=dvd(x,i); v=dvv(x,i); c=mpush(q,d,v);
  if(p==Cast::undefined) p=c;
 }
 return p;  // return module type of overall parent container
}

static K mdv(K x,J n,Kmodule *l=nullptr,J d=0,K v=nullptr); // higher-level call, can add to existing module
static K mdv(K x,J n,Kmodule *l,J d,K v) {
 Cast c; Layers q; if(l) mstack(l,q);
 c=v ? mpush(q,d ? d : q.size(),v) : mdv(x,n,q);
 return l ? (K)0 : rootlayer(c,q);
}

static Cast mtable(K x,Layers &q) { // process table/dict w'depth,layer,options,parms,buffers
 Cast c,p=Cast::undefined; J j=0,n=x->t==99 ? 1 : xlen(x);
 for(J i=0;i<n;++i) {
  std::tie(c,j)=mpush(q, j, statedepth(x,i),   statemodule(x,i), statename(x,i),
                            stateoptions(x,i), stateparms(x,i),  statebuffers(x,i));
  if(p==Cast::undefined) p=c;
 }
 return p;
}

static K mtable(K x,Kmodule *l=nullptr);  //higher-level call, can also add to existing module if supplied
static K mtable(K x,Kmodule *l) {Layers q; if(l) mstack(l,q); Cast c=mtable(x,q); return l ? (K)0 : rootlayer(c,q);}

static void mextend(Layer& a,Cast c,J d,Layers& q) {
 if(d) mdepth(c,d,q);
 if(container(c))
  addparent(a,q);
 else
  addchild(a,q);
}

static void mextend(Kmodule *x,Kmodule *y,J d=0);
static void mextend(Kmodule *x,Kmodule *y,J d) {Layers q; mstack(x,q); mextend(y->m,y->c,d ? d : q.size(),q);}

// --------------------------------------------------------------------------------------------
// mget - extract module options and, optionally, parameters & buffers to k array
// --------------------------------------------------------------------------------------------
void mget(bool a,int64_t d,const char* s,bool t,const Module& m,K x) {
 Cast c; K o,*k=kK(x); std::tie(c,o)=mopt(a,m);
 if(!s) s="";
 if(t) {
  ja(&k[0], &d);
  js(&k[1], msym(c));
  js(&k[2], cs(s));
  jk(&k[3], o);
  if(x->n == 6)
   jk(&k[4], kget(m.named_parameters(false))),
   jk(&k[5], kget(m.named_buffers(false)));
  for(auto& i:m.named_children())
   mget(a,d+1,i.key().c_str(),t,*i.value(),x);
 } else {
  TORCH_CHECK(!m.children().size(), msym(c), ": unexpected child module(s)");
  k[0]=kj(d);
  k[1]=ks(msym(c));
  k[2]=ks(cs(s));
  k[3]=o;
  if(x->n == 6)
   k[4]=kget(m.named_parameters(false)),
   k[5]=kget(m.named_buffers(false));
 }
}

K mget(bool a,bool b,const Module& m) {  
// a-true for all options else non-defaults, b-true for full state w'parms & buffers, s-name
 K k=mkeys(b),v=ktn( 0, b ? 6 : 4);  // key,val for depth,module,name,options w'parms & buffers if b
 if(container(m) || m.children().size()) {
  for(J i=0; i<v->n; ++i) kK(v)[i]=ktn(!i ? KJ : (i<3 ? KS : 0), 0);
  mget(a,0,mname(m),true,m,v);
  return xT(xD(k,v));
 } else {
  mget(a,0,mname(m),false,m,v);
  return xD(k,v);
 }
}

// ------------------------------------------------------------------------------------------
//  main api function defined in k
// ------------------------------------------------------------------------------------------
KAPI module(K x) {
 KTRY
  bool a=env().alloptions; J d,n; Kmodule *l,*g; Kmodel *m;
  if((l=xmodule(x)) || (l=xmodule(x,0))) {       // allocated module ptr supplied
   if(x->n==1 || (x->n==2 && xbool(x,1,a))) {    // no other args or boolean flag
    return mget(a,false,mref(l->m));             // return module options
   } else if(x->n==2) {                          // else if allocated module & non-boolean arg
    if((g=xmodule(x,1)))                         // if another allocated module
     return mextend(l,g), kfree(x,1), (K)0;      // add to last container module in chain
    else if((n=xdv(x,1)))                        // 2nd arg of depth,value pair(s)
     return mdv(kK(x)[1],n,l);                   // add module(s) specified in depth,value pair(s)
    else if(xstate(x,1))                         // if state dictionary/table detected as 2nd arg
     return mtable(kK(x)[1],l);                  // add definition(s) to existing module(s)
    else                                         // fallback: assume 2nd arg is nested tree spec
     return mtree(kK(x)[1],0,l);                 // add module(s) to last container in existing module
   } else if(x->n==3 && xlong(x,1,d)) {          // else if allocated module & depth given w'3rd arg
    if((g=xmodule(x,2)))                          // if another allocated module
     return mextend(l,g,d), kfree(x,2), (K)0;    // add module at given depth in chain
    else
     return mdv(nullptr,0,l,d,kK(x)[2]);         // add single module definition at indicated depth
   } else {
    AT_ERROR("module: ", mlabel(mref(l->m)), " given as 1st arg, but unable to parse remaining arg(s)");
   }
  } else if(xstate(x)) {                         // module table or dictionary supplied
   return mtable(x);
  } else if((m=xmodel(x))) {                     // model ptr supplied, extract module with added reference
   return kmodule(m->mc,m->m);
  } else if((n=xdv(x))) {                        // depth-value pairs supplied
   return mdv(x,n);
  } else {
   return mtree(x);                              // nested tree representation
  }
 KCATCH("module");
}

K modulehelp(Cast c) {
 switch(c) {
  case Cast::adaptavg1d:      return adapt(nn::AdaptiveAvgPool1dOptions(3));
  case Cast::adaptavg2d:      return adapt(nn::AdaptiveAvgPool2dOptions({3,2}));
  case Cast::adaptavg3d:      return adapt(nn::AdaptiveAvgPool3dOptions({3,2,4}));
  case Cast::adaptmax1d:      return adapt(nn::AdaptiveMaxPool1dOptions(3));
  case Cast::adaptmax2d:      return adapt(nn::AdaptiveMaxPool2dOptions({3,2}));
  case Cast::adaptmax3d:      return adapt(nn::AdaptiveMaxPool3dOptions({3,2,4}));
  case Cast::adrop:           return drop(true,nn::AlphaDropoutOptions());
  case Cast::attention:       return attention(true,nn::MultiheadAttentionOptions(2048,8));
  case Cast::avgpool1d:       return avgpool(true,nn::AvgPool1dOptions(3));
  case Cast::avgpool2d:       return avgpool(true,nn::AvgPool2dOptions({3,2}));
  case Cast::avgpool3d:       return avgpool(true,nn::AvgPool3dOptions({3,2,2}));
  case Cast::batchnorm1d:
  case Cast::batchnorm2d:
  case Cast::batchnorm3d:     return batchnorm(true,nn::BatchNormOptions(32));
  case Cast::bilinear:        return bilinear(true,nn::BilinearOptions(20,30,40));
  case Cast::cat:             return dim(true,c,CatOptions().dim());
  case Cast::celu:            return alpha(true,nn::CELUOptions());
  case Cast::conv1d:          return conv(true,nn::detail::ConvNdOptions<1>(16,32,3));
  case Cast::conv2d:          return conv(true,nn::detail::ConvNdOptions<2>(16,32,{3,5}));
  case Cast::conv3d:          return conv(true,nn::detail::ConvNdOptions<3>(16,32,{3,5,2}));
  case Cast::convtranspose1d: return conv(true,nn::detail::ConvNdOptions<1>(128,64,5).transposed(true));
  case Cast::convtranspose2d: return conv(true,nn::detail::ConvNdOptions<2>(128,64,{3,5}).transposed(true));
  case Cast::convtranspose3d: return conv(true,nn::detail::ConvNdOptions<3>(128,64,{3,5,2}).transposed(true));
  case Cast::crossmap2d:      return localnorm(true,c,nn::CrossMapLRN2dOptions(2));
  case Cast::decoder:         return decoder(true,nn::TransformerDecoderOptions(
                                             nn::TransformerDecoderLayerOptions(512,8),6)
                                             .norm(AnyModule(nn::LayerNorm(nn::LayerNormOptions({512})))));
  case Cast::decoderlayer:    return codelayer(true,nn::TransformerDecoderLayerOptions(512,8));
  case Cast::drop:            return drop(true,nn::DropoutOptions());
  case Cast::drop2d:          return drop(true,nn::Dropout2dOptions());
  case Cast::drop3d:          return drop(true,nn::Dropout3dOptions());
  case Cast::elu:             return alpha(true,nn::ELUOptions());
  case Cast::embed:           return embed(true,c,nn::EmbeddingOptions(1000,64),{});
  case Cast::embedbag:        return embed(true,c,nn::EmbeddingBagOptions(1000,64),{});
  case Cast::encoder:         return encoder(true,nn::TransformerEncoderOptions(
                                             nn::TransformerEncoderLayerOptions(512,8),6)
                                             .norm(AnyModule(nn::LayerNorm(nn::LayerNormOptions({512})))));
  case Cast::encoderlayer:    return codelayer(true,nn::TransformerEncoderLayerOptions(512,8));


  case Cast::expand:          return getsize(true,SizeOptions({-1,-1,28,28}));
  case Cast::fadrop:          return drop(true,nn::FeatureAlphaDropoutOptions());
  case Cast::flatten:         return flatten(true,nn::FlattenOptions());
  case Cast::fmaxpool2d:      return fpool(true,nn::FractionalMaxPool2dOptions({2,4})  .output_size(ExpandingArray<2>({16,32})));
  case Cast::fmaxpool3d:      return fpool(true,nn::FractionalMaxPool3dOptions({2,4,3}).output_size(ExpandingArray<3>({16,32,24})));
  case Cast::fold:            return fold(true,nn::FoldOptions({4,6},{2,3}));
  case Cast::gelu:            return KDICT;
  case Cast::glu:             return dim(true,c,nn::GLUOptions().dim());
  case Cast::groupnorm:       return groupnorm(true,nn::GroupNormOptions(3,6));
  case Cast::gru:             return rnn(true,nn::GRUOptions(10,20));
  case Cast::hardshrink:      return lambda(true,c,torch::nn::HardshrinkOptions().lambda());
  case Cast::hardtanh:        return hardtanh(true,nn::HardtanhOptions());
  case Cast::identity:        return KDICT;
  case Cast::instancenorm1d:
  case Cast::instancenorm2d:
  case Cast::instancenorm3d:  return batchnorm(true,nn::InstanceNormOptions(100));
//case Cast::interpolate:
  case Cast::layernorm:       return layernorm(true,nn::LayerNormOptions({32,10}));
  case Cast::leakyrelu:       return slope(true,c,nn::LeakyReLUOptions());
  case Cast::linear:          return linear(true,nn::LinearOptions(784,10));
  case Cast::localnorm:       return localnorm(true,c,nn::LocalResponseNormOptions(2));
  case Cast::logsigmoid:      return KDICT;
  case Cast::logsoftmax:      return dim(true,c,nn::LogSoftmaxOptions(1).dim());
  case Cast::lppool1d:        return lppool(true,nn::LPPool1dOptions(2,3));
  case Cast::lppool2d:        return lppool(true,nn::LPPool2dOptions(1.2,{2,3}));
  case Cast::lstm:            return rnn(true,nn::LSTMOptions(10,20));
  case Cast::maxpool1d:       return maxpool(true,nn::MaxPool2dOptions(3));
  case Cast::maxpool2d:       return maxpool(true,nn::MaxPool2dOptions({3,2}));
  case Cast::maxpool3d:       return maxpool(true,nn::MaxPool3dOptions({3,2,2}));
  case Cast::modulelist:      return KDICT;
  case Cast::mul:             return KDICT;
//case Cast::normalize:       

  case Cast::pad1d:           return cpad(nn::ConstantPad1dOptions({1,2},0));
  case Cast::pad2d:           return cpad(nn::ConstantPad2dOptions({1,1,2,2},0));
  case Cast::pad3d:           return cpad(nn::ConstantPad3dOptions({3,3,6,6,0,1}, 3.5));

  case Cast::sigmoid:         return KDICT;
  case Cast::reshape:         return getsize(true,SizeOptions({-1,1,28,28}));
  case Cast::rnn:             return rnn(true,nn::RNNOptions(10,20));
  case Cast::softmax:         return dim(true,c,nn::SoftmaxOptions(1).dim());
  case Cast::softmax2d:       return KDICT;
  case Cast::softmin:         return dim(true,c,nn::SoftminOptions(1).dim());
  case Cast::softsign:        return KDICT;
  case Cast::tanh:            return KDICT;
  case Cast::tanhshrink:      return KDICT;

/*
 } else if(auto* m=g.as<nn::ReLU>())  { c=Cast::relu;  x=inplace(a,m->options.inplace());
 } else if(auto* m=g.as<nn::SELU>())  { c=Cast::selu;  x=inplace(a,m->options.inplace());
 } else if(auto* m=g.as<nn::ReLU6>()) { c=Cast::relu6; x=inplace(a,m->options.inplace());

 } else if(auto* m=g.as<Squeeze>())    { c=Cast::squeeze;    x=squeeze(a,m->options);
 } else if(auto* m=g.as<Unsqueeze>())  { c=Cast::unsqueeze;  x=squeeze(a,m->options);

 } else if(auto* m=g.as<nn::Transformer>())             { c=Cast::transformer;  x=transformer(a,m->options);
*/

  default: AT_ERROR("nyi");
 }
}

// ----------------------------------
// module fns defined in k namespace
// ----------------------------------
void nnfn(K x) {
 fn(x, "seq",         KFN(seq),          1);    // convenience fn for sequential layers
 fn(x, "module",      KFN(module),       1);    // api function for layer create/query
 fn(x, "adaptavg1d",  KFN(adaptavg1d),   1);    // functional form of modules/activations
 fn(x, "adaptavg2d",  KFN(adaptavg2d),   1);
 fn(x, "adaptavg3d",  KFN(adaptavg3d),   1);
 fn(x, "adaptmax1d",  KFN(adaptmax1d),   1);
 fn(x, "adaptmax2d",  KFN(adaptmax2d),   1);
 fn(x, "adaptmax3d",  KFN(adaptmax3d),   1);
 fn(x, "fmaxpool2d",  KFN(fmaxpool2d),   1);
 fn(x, "fmaxpool3d",  KFN(fmaxpool3d),   1);
 fn(x, "avgpool1d",   KFN(avgpool1d),    1);
 fn(x, "avgpool2d",   KFN(avgpool2d),    1);
 fn(x, "avgpool3d",   KFN(avgpool3d),    1);
 fn(x, "pad",         KFN(kpad),         1);
 fn(x, "celu",        KFN(celu),         1);
 fn(x, "elu",         KFN(elu),          1);
 fn(x, "flatten",     KFN(kflatten),     1);
 fn(x, "fold",        KFN(Fold),         1);
 fn(x, "glu",         KFN(glu),          1);
 fn(x, "hardshrink",  KFN(hardshrink),   1);
 fn(x, "hardtanh",    KFN(Hardtanh),     1);
 fn(x, "interpolate", KFN(kinterpolate), 1);
 fn(x, "leakyrelu",   KFN(leakyrelu),    1);
 fn(x, "linear",      KFN(Linear),       1);
 fn(x, "bilinear",    KFN(Bilinear),     1);
 fn(x, "logsigmoid",  KFN(logsigmoid),   1);
 fn(x, "logsoftmax",  KFN(logsoftmax),   1);
 fn(x, "lppool1d",    KFN(lppool1d),     1);
 fn(x, "lppool2d",    KFN(lppool2d),     1);
 fn(x, "maxpool1d",   KFN(maxpool1d),    1);
 fn(x, "maxpool2d",   KFN(maxpool2d),    1);
 fn(x, "maxpool3d",   KFN(maxpool3d),    1);
 fn(x, "normalize",   KFN(Normalize),    1);
 fn(x, "prelu",       KFN(Prelu),        1);
 fn(x, "gelu",        KFN(gelu),         1);
 fn(x, "relu",        KFN(relu),         1);
 fn(x, "relu6",       KFN(relu6),        1);
 fn(x, "rrelu",       KFN(Rrelu),        1);
 fn(x, "selu",        KFN(selu),         1);
 fn(x, "softmax",     KFN(softmax),      1);
 fn(x, "softmin",     KFN(softmin),      1);
 fn(x, "softplus",    KFN(Softplus),     1);
 fn(x, "softsign",    KFN(softsign),     1);
 fn(x, "softshrink",  KFN(softshrink),   1);
 fn(x, "tanhshrink",  KFN(tanhshrink),   1);
 fn(x, "threshold",   KFN(Threshold),    1);
 fn(x, "unfold",      KFN(Unfold),       1);
 fn(x, "pairwise",    KFN(Pairwise),     1);
 fn(x, "pdist",       KFN(pdist),        1);
 fn(x, "similar",     KFN(Similar),      1);
}

/*
normalize, interpolate  -- functional form implemented, add module?
pairwise distance & cosine similarity: in both module & functional form but forward method needs 2 input tensors
fractional pool -- try with indices registered as buffer?
embeddingbag -- forward w'defaults should work with sequential
multi-head attention -- not in 1.4, wait for patch or 1.5
1.7 adds SiLU, UnFlatten, transformer modules..
*/
