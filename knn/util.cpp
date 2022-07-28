#include "../ktorch.h"
#include "util.h"

namespace knn {

// -----------------------------------------------------------------
// maxargs - check for maximum no. of tensor args in forward calc
// -----------------------------------------------------------------
size_t maxargs(const AnyModule& m,const char *c) {
 auto h=m.type_info().hash_code();
 for(const auto& a:env().modules) {
  if(std::get<2>(a)==h) {
   for(auto t:std::get<8>(a))
    TORCH_CHECK(t==Arg::tensor, c,": ",mlabel(m.type_info().name())," layer uses non-tensor arg ",argname(t));
   return std::get<7>(a);
  }
 }
 TORCH_ERROR(c,": unable to determine number of tensor args for ",mlabel(m.type_info().name())," layer");
}

// -----------------------------------------------------------------
// code  - check args for symbol,  else error w'module & option name
// -----------------------------------------------------------------
S code(K x,J i,Cast c,Setting s) {
 S m;
 TORCH_CHECK(xsym(x,i,m), msym(c)," ",mset(s),": expected symbol, given ",kname(x,i));
 return m;
}

S code(const Pairs& p,Cast c) {
 TORCH_CHECK(p.t==-KS, msym(c)," ",p.k,": expected symbol, given ",kname(p.t));
 return p.s;
}

// ------------------------------------------------------------------
// otype - check args for optional data type (null symbol -> nullopt)
// ------------------------------------------------------------------
c10::optional<Dtype> otype(S s) {if(nullsym(s)) return c10::nullopt; else return stype(s);}
c10::optional<Dtype> otype(K x,J i,Cast c,Setting s) {return otype(code(x,i,c,s));}
c10::optional<Dtype> otype(const Pairs& p,Cast c)    {return otype(code(p,c));}

// -----------------------------------------------------------------------------------
// mpos - throw error if too many positional arguments
// mpair - throw error if unrecognized name in name-value pairs
// -----------------------------------------------------------------------------------
void mpos(K x,Cast c,J n) {
 TORCH_ERROR(msym(c),": expecting up to ",n," positional args, ",xlen(x)," given");
}

void mpair(Cast c,const Pairs& p) {
 TORCH_ERROR(msym(c)," option: ",p.k," not recognized");
}

// ----------------------------------------------------------------------------
// mset - map to/from setting to symbol, w'optional module type if given symbol
// msetting - add to dictionary of module settings given enum & k value
// ----------------------------------------------------------------------------
S mset(Setting x) {
 for(auto& m:env().mset) if(x == std::get<1>(m)) return std::get<0>(m);
 TORCH_ERROR("unrecognized module option: ",(I)x);
}

Setting mset(S x,Cast c) {
 for(const auto& m:env().mset) if(x == std::get<0>(m)) return std::get<1>(m);
 if(c == Cast::undefined)
  TORCH_ERROR("unrecognized option: `",x);
 else
  TORCH_ERROR(msym(c),": unrecognized option `",x);
}

void msetting(K x,Setting s,K v) {dictadd(x,mset(s),v);}

// ----------------------------------------------------------------------------------------------------
// mbool - check args for boolean, else error w'module & option name
// ----------------------------------------------------------------------------------------------------
bool mbool(K x,J i,Cast c,Setting s) {
 bool b;
 TORCH_CHECK(xbool(x,i,b), msym(c)," ",mset(s),": expected boolean scalar, given ",kname(x,i));
 return b;
}

bool mbool(const Pairs& p,Cast c) {
 TORCH_CHECK(p.t==-KB, msym(c)," ",p.k,": expected boolean scalar, given ",kname(p.t));
 return p.b;
}

// --------------------------------------------------------------------
// int64 - check args for long int, else error w'module & option
// int64n - int64 but returns optional, i.e. nullopt if k value is null
// --------------------------------------------------------------------
int64_t int64(K x,J i,Cast c,Setting s) {
 int64_t n;
 TORCH_CHECK(xint64(x,i,n), msym(c)," ",mset(s),": expected long scalar, given ",kname(x,i));
 return n;
}

int64_t int64(const Pairs& p,Cast c) {
 TORCH_CHECK(p.t==-KJ, msym(c)," ",p.k,": expected long scalar, given ",kname(p.t));
 return p.j;
}

c10::optional<int64_t> int64n(K x,J i,Cast c,Setting s) {auto n=int64(x,i,c,s); if(null(n)) return c10::nullopt; else return n;}
c10::optional<int64_t> int64n(const Pairs& p,Cast c)    {auto n=int64(p,c);     if(null(n)) return c10::nullopt; else return n;}

// ----------------------------------------------------------------------
// mdouble - check for double(or long) from positional or name-value pair
// optdouble - call mdouble() but return null if k null supplied
// ----------------------------------------------------------------------
double mdouble(K x,J i,Cast c,Setting s) {
 double f;
 TORCH_CHECK(xnum(x,i,f), msym(c)," ",mset(s),": expected double, given ",kname(x,i));
 return f;
}

double mdouble(const Pairs& p,Cast c) {
 TORCH_CHECK(p.t==-KJ || p.t==-KF, msym(c)," ",p.k,": expected double, given ",kname(p.t));
 return pdouble(p);
}

c10::optional<double> optdouble(K x,J i,Cast c,Setting s) {double d=mdouble(x,i,c,s); if(d==d) return d; else return c10::nullopt;}
c10::optional<double> optdouble(const Pairs& p,Cast c)    {double d=mdouble(p,c);     if(d==d) return d; else return c10::nullopt;}

// ----------------------------------------------------------------------------------------
// mlongs - check for long(s), return vector else error specific to module and setting
// ltensor - define tensor from long(s), else error specific to module & setting
// ftensor - define tensor from long/float/double(s), else error specific to setting
// mdoubles - check for double(s), return vector else error specific to module and setting
// ----------------------------------------------------------------------------------------
LongVector mlongs(K x,J i,Cast c,Setting s) {
 IntArrayRef a;
 TORCH_CHECK(xsize(x,i,a), msym(c)," ",mset(s),": expected long(s), given ",kname(x,i));
 return a.vec();
}

LongVector mlongs(const Pairs& p,Cast c) {
 IntArrayRef a;
 TORCH_CHECK(p.t==-KJ || p.t==KJ, msym(c)," ",p.k,": expected long(s), given ",kname(p.t));
 psize(p,a);
 return a.vec();
}

Tensor ltensor(K x,J i,Cast c,Setting s) {
 Tensor t; if(!xten(x,i,t)) t=kput(x,i);
 TORCH_CHECK(t.dtype()==torch::kLong, msym(c)," ",mset(s),": long(s) expected, given ",t.dtype(),"(s)");
 return t;
}

Tensor ltensor(const Pairs& p,Cast c) {
 Tensor t; pten(p,t);
 TORCH_CHECK(t.dtype()==torch::kLong, msym(c)," ",p.k,": long(s) expected, given ",t.dtype(),"(s)");
 return t;
}

Tensor ftensor(K x,J i,Cast c,Setting s) {
 Tensor t; if(!xten(x,i,t)) t=kput(x,i); if(t.dtype()==torch::kLong) t=t.to(torch::kDouble);
 TORCH_CHECK(t.is_floating_point(), msym(c)," ",mset(s),": double(s) expected, given ",t.dtype(),"(s)");
 return t;
}

Tensor ftensor(const Pairs& p,Cast c) {
 Tensor t; pten(p,t); if(t.dtype()==torch::kLong) t=t.to(torch::kDouble);
 TORCH_CHECK(t.is_floating_point(), msym(c)," ",p.k,": double(s) expected, given ",t.dtype(),"(s)");
 return t;
}

DoubleVector mdoubles(K x,J i,Cast c,Setting s) {
 J n; F *f; IntArrayRef a; DoubleVector v;
 if(xsize(x,i,a)) {
  for(const auto j:a) v.push_back(j);
 } else if(xdouble(x,i,n,f)) {
  v=DoubleArrayRef(f,n).vec();
 } else {
  TORCH_ERROR(msym(c)," ",mset(s),": expected double(s), given ",kname(x,i));
 }
 return v;
}

DoubleVector mdoubles(const Pairs& p,Cast c) {
 DoubleVector v;
 if(p.t==-KJ || p.t==KJ) {
  IntArrayRef a; psize(p,a);
  for(const auto j:a) v.push_back(j);
 } else if(p.t==-KF || p.t==KF) {
  DoubleArrayRef a; pdoubles(p,a); v=a.vec();
 } else {
  TORCH_ERROR(msym(c)," ",p.k,": expected double(s), given ",kname(p.t));
 }
 return v;
}

} // namespace knn
