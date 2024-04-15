#include "ktorch.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

// --------------------------------------------------------------------------------------------------
// krrbuf - copy msg to a buffer for signalling error to k
// dictadd - add an entry in a dictionary mapping symbol -> k value
// xind - true if i is valid index of k list (type=0)
// kptr - given void *, add to pointer list & return k list of one long scalar = (intptr_t)void *
// ptrtype - true if x is a 1-element list of a single scalar
// ptrflag - true if x is found in map of pointers
// mapped - true if ptr type and actively mapped
// xptr - given k value, return true if enclosed scalar and in set of maintained pointers
// xtag - if enclosed integer ptr detected from k, return pointer to tag structure
// --------------------------------------------------------------------------------------------------
S krrbuf(const char *s) {
 static C thread_local b[4096]; b[0]=0; 
 return strncat(b, s, sizeof(b)-1);
}

void dictadd(K x,       char* s, K v){K *k=kK(x); js(&k[0],cs(s)); jk(&k[1],v);}
void dictadd(K x, const char* s, K v){K *k=kK(x); js(&k[0],cs(s)); jk(&k[1],v);}

bool xind(K x,J i) {return !x->t && -1<i && i<x->n;}
bool xind(K x,J i,Ktype k) {return x->t==k && -1<i && i<x->n;}
K kptr(void *v){J j=(intptr_t)v; pointer().insert(j); return knk(1,kj(j));}

bool ptrtype(K x) {return !x->t && x->n==1 && kK(x)[0]->t==-KJ;}
bool ptrflag(K x) {return pointer().count(kK(x)[0]->j);}
bool  mapped(K x) {return ptrtype(x) && ptrflag(x);}

bool xptr(K x) {
 if(ptrtype(x)) {
  TORCH_CHECK(ptrflag(x), "stale pointer");
  return true;
 } else {
  return false;
 }
}

bool xptr(K x,J i) {return xind(x,i) && xptr(kK(x)[i]);}

Ktag* xtag(K x) {return xptr(x) ? (Ktag*)kK(x)[0]->j : nullptr;}
Ktag* xtag(K x,J i) {return xind(x,i) ? xtag(kK(x)[i]) : nullptr;}

// ---------------------------------------------------------------------------------
// null - true if null for given type
// match - return true if scalars match (check long/double value)
// kscalar - return k double/long from torch scalar
// resolve - serialize, then deserialize, to remove k object types created via c-api
// resolvedict - return dict with simple value list if general list of all same type
// Resolve - k api function to call `resolve`
// ---------------------------------------------------------------------------------
bool null(const char* x) { return !x || !strlen(x);}
bool null(const J x)     { return x == nj; }

bool match(const Scalar &x,const Scalar &y) {
 if(x.isIntegral(false)) {
  if(y.isIntegral(false))
   return x.toLong() == y.toLong();
  else if(y.isFloatingPoint())
   return x.toDouble() == y.toDouble();
 } else if(x.isFloatingPoint()) {
  if(y.isFloatingPoint() || y.isIntegral(false))
   return x.toDouble() == y.toDouble();
 }
 TORCH_ERROR("unexpected scalar type(s), neither integral or floating point, cannot compare");
}

K kscalar(const Scalar &s) {
 if(s.isIntegral(false))
  return kj(s.toLong());
 else if(s.isFloatingPoint())
  return kf(s.toDouble());
 TORCH_ERROR("unexpected scalar type(s), neither integral or floating point, cannot convert");
}

K resolve(K x) {
 K b=b9(-1,x);
 TORCH_CHECK(b, "b9: failed to serialize ",kname(x));
 K z=d9(b); r0(b);
 TORCH_CHECK(z, "d9: failed to deserialize ",kname(x));
 return z;
}

K resolvedict(K x) {
 K y=kK(x)[1]; auto t=(!y->t && y->n) ? kK(y)[0]->t : 0;
 if(t==-KB || t==-KJ || t==-KS || t==-KF) {
  for(J i=1; i<y->n; ++i)
   if(t != kK(y)[i]->t) return x;
  K v=ktn(-t,y->n);
  for(J i=0; i<y->n; ++i) {
   switch(t) {
    case -KB: kG(v)[i]=kK(y)[i]->g; break;
    case -KJ: kJ(v)[i]=kK(y)[i]->j; break;
    case -KS: kS(v)[i]=kK(y)[i]->s; break;
    case -KF: kF(v)[i]=kK(y)[i]->f; break;
   }
  }
  kK(x)[1]=v; r0(y);
 }
 return x;
}

KAPI Resolve(K x) {
 KTRY
  return resolve(x);
 KCATCH("resolve");
}

// ------------------------------------------------------------------------------------------
// xlen - 1 if scalar else x->n for lists, no. of table rows or dictionary elements
// mapclass - map class enum to symbol
// kname - string from k data type or object
// ------------------------------------------------------------------------------------------
J xlen(K x) {
 if(x->t < 0 || x->t > 99) return 1;
 else if(x->t < 98)        return x->n;
 else if(x->t == 98)       return xlen(kK(kK(x->k)[1])[0]);
 else                      return xlen(kK(x)[0]);
}

J xlen(K x,J i) {return xind(x,i) ? xlen(kK(x)[i]) : -1;}

S mapclass(Class a) {
 for(const auto& m:env().kclass)
  if(a==std::get<1>(m)) return std::get<0>(m);
 TORCH_ERROR("unrecognized class: ", (I)a);
}

const char* kname(Ktype k) {
 Ktype t=abs(k); bool b=k<0;
 switch(t) {
  case 0: return "general list";
  case 1: return b ? "boolean scalar" : "boolean list";
  case 2: return b ? "guid scalar" : "guid list";
  case 4: return b ? "byte scalar" : "byte list";
  case 5: return b ? "short scalar" : "short list";
  case 6: return b ? "int scalar" : "int list";
  case 7: return b ? "long scalar" : "long list";
  case 8: return b ? "real scalar" : "real list";
  case 9: return b ? "double scalar" : "double list";
  case 10: return b ? "char scalar" : "char list";
  case 11: return b ? "symbol scalar" : "symbol list";
  case 12: return b ? "timestamp scalar" : "timestamp list";
  case 13: return b ? "month scalar" : "month list";
  case 14: return b ? "date scalar" : "date list";
  case 15: return b ? "datetime scalar" : "datetime list";
  case 16: return b ? "timespan scalar" : "timespan list";
  case 17: return b ? "minute scalar" : "minute list";
  case 18: return b ? "second scalar" : "second list";
  case 19: return b ? "time scalar" : "time list";
  case 97: return "nested sym enum";
  case 98: return "table";
  case 99: return "kdictionary";
  case 100: return "lambda";
  case 101: return "null/unary primitive";
  case 102: return "operator";
  case 103: return "adverb";
  case 104: return "projection";
  case 105: return "composition";
  case 106: return "f'";
  case 107: return "f/";
  case 108: return "f\\";
  case 109: return "f':";
  case 110: return "f/:";
  case 111: return "f\\:";
  case 112: return "dynamic load fn";
  case -128: return "error";
  default:
    if(t>19 && t<77)
     return b ? "enum scalar" : "enum list";
    else if(t>76 && t<97)
     return "map";
    else
     return "unrecognized type";
 }
}

const char* kname(K x) {return xptr(x) ? mapclass(xtag(x)->a) : kname(x->t);}
const char* kname(K x,J i) {return xind(x,i) ? kname(kK(x)[i]) : kname(x);}
 
// ----------------------------------------------------------------------------
// ksizeof - given k type, return size of element, e.g. KF -> 8
// maptype - map k data type to/from torch type
// mapattr - make attr enum to symbol
// emap - map sym -> enum (enum used to pick variant member, e.g. torch::kMean)
// input,output - map from Input/Output variant value to symbol
// ----------------------------------------------------------------------------
J ksizeof(Ktype k) {
 switch(k) {
  case KE:     return sizeof(E);
  case KF:     return sizeof(double);
  case KJ:     return sizeof(J);
  case KI:     return sizeof(I);
  case KSHORT: return sizeof(H);
  case KC:     return sizeof(C);
  case KB:
  case KG:     return sizeof(G);
  default: TORCH_ERROR("no element size for k ",kname(k)); return -1;
 }
}

Ktype maptype(TypeMeta s) {
 for(const auto& m:env().dtype)
  if(s==std::get<1>(m)) return std::get<2>(m);
 TORCH_ERROR("no k data type found for torch type: ",s);
 return 0;
}

TypeMeta maptype(Ktype k) {
 Ktype t=(k<0) ? -k : k;
 for(const auto &m:env().ktype)
  if(t==std::get<0>(m)) return std::get<1>(m);
 TORCH_ERROR("no torch type found for k: ",kname(k));
}

S mapattr(Attr a) {
 for(const auto& m:env().attr)
  if(a==std::get<1>(m)) return std::get<0>(m);
 TORCH_ERROR("unrecognized attribute: ", (I)a);
}

Enum emap(S s) {
 for(const auto& m:env().enums)
  if(std::get<0>(m)==s) return std::get<1>(m);
 return Enum::undefined;
}

S emap(Enum e) {
 for(const auto& m:env().enums)
  if(std::get<1>(m)==e) return std::get<0>(m);
 TORCH_ERROR("unrecognized enumeration: ",(I)e);
}

S  inputname(const  Input& x) {return env().in [x.index()];}
S outputname(const Output& x) {return env().out[x.index()];}

// ------------------------------------------------------------------------------------------
// statekey - map from state attribute enumeration to symbol, e.g. State::parms -> `parms
// statefind - search dictionary keys/table colums for symbol matching given enumeration
// statelong - search dict/table for long value
// statesym - given dict/table defining module(s), find symbols for module else null
// statedict - given enumeration, return k dictionary stored at matching key/col else null
// ------------------------------------------------------------------------------------------
S statekey(State e) {
 for(const auto& m:env().state)if(e==std::get<1>(m)) return std::get<0>(m);
 TORCH_ERROR("unrecognized state attribute: ",(I)e);
}

J statefind(State e,K x,bool r) {
 if(!xstate(x))
  TORCH_ERROR("expected dictionary or table describing state, given ",kname(x));
 S s=statekey(e); K k=kK(x->t == 98 ? x->k : x)[0];
 for(J i=0;i<k->n;++i) if(kS(k)[i]==s) return i;
 TORCH_CHECK(!r, "unable to find state attribute: ",statekey(e));
 return -1;
}

static J statelong(State e,bool r,K x,J j) { //e:enum, e.g. State::depth, r:required flag, x:dict/table, j:table row
 J i=statefind(e,x,r);
 if(i<0) {
  return nj;
 } else if(x->t==99) {
  K v=kK(x)[1];
  if(v->t) {
   TORCH_CHECK(v->t==KJ, statekey(e),": expected long, given ",kname(v));
   return kJ(v)[i];
  } else {
   v=kK(v)[i];
   TORCH_CHECK(v->t==-KJ, statekey(e),": expected long, given ",kname(v));
   return v->j;
  }
 } else if(x->t==98) {
  K v=kK(kK(x->k)[1])[i];
  TORCH_CHECK(v->t==KJ, statekey(e),": expected long, given ",kname(v));
  TORCH_CHECK(j>-1 && j<v->n, statekey(e),"[", j,"] index beyond ",v->n,"-row table");
  return kJ(v)[j];
 } else {
  TORCH_ERROR("expecting state dictionary or table, given ",kname(x));
 }
}

S statesym(State e, bool r,K x,J j) {
// e:enum, e.g. State::module, r:required, x:dict/table, j:table row or -1
 J i=statefind(e,x,r);
 if(i<0) {
  return nullptr;
 } else if(x->t==99) {
  K v=kK(x)[1];
  if(v->t) {
   TORCH_CHECK(v->t==KS, statekey(e),": expected symbol, given ",kname(v));
   return kS(v)[i];
  } else {
   v=kK(v)[i];
   TORCH_CHECK(v->t==-KS, statekey(e),": expected symbol, given ",kname(v));
   return v->s;
  }
 } else if(x->t==98) {
  K v=kK(kK(x->k)[1])[i];
  TORCH_CHECK(v->t==KS, statekey(e),": expected symbol, given ",kname(v));
  TORCH_CHECK(j>-1 && j<v->n, statekey(e),"[", j,"] index beyond ",v->n,"-row table");
  return kS(v)[j];
 } else {
  TORCH_ERROR("expecting state dictionary or table, given ",kname(x));
 }
}

K statedict(State e,K x,J j) {
// e:enum, e.g. State::options, x:dict/table, j:row (if table, else -1)
 J i=statefind(e,x);
 if(i<0) return nullptr;
 K v=x->t == 98 ? kK(kK(x->k)[1])[i] : kK(x)[1];
 if(x->t == 99) j=i;
 TORCH_CHECK(!v->t, statekey(e),": expected general list, given ",kname(v));
 TORCH_CHECK(-1<j && j<v->n, statekey(e),"[",j,"] index beyond ",v->n,"-row table");
 v=kK(v)[j];
 TORCH_CHECK(v->t==99, statekey(e),": expected dictionary, given ",kname(v));
 return v;
}

static K statelist(State e,K x,J j) {  // e:enum, e.g. State::size, x:dict/table, j:row (if table)
 J i=statefind(e,x);
 if(i<0) return nullptr;
 K v=x->t == 98 ? kK(kK(x->k)[1])[i] : kK(x)[1];
 if(x->t == 99) j=i;
 TORCH_CHECK(!v->t, statekey(e),": expected general list, given ",kname(v));
 TORCH_CHECK(-1<j && j<v->n, statekey(e),"[",j,"] index beyond ",v->n,"-row table");
 v=kK(v)[j];
 return v;
}

K statetable(State e,K x) {
 J i=statefind(e,x);
 if(i<0) return nullptr;
 TORCH_CHECK(x->t==99, "expecting dictionary containing ",statekey(e),", given ",kname(x->t));
 K v=kK(x)[1]; TORCH_CHECK(!v->t, statekey(e),": expected general list, given ",kname(v));
 v=kK(v)[i]; TORCH_CHECK(v->t==98, statekey(e),": expected table, given ",kname(v));
 return v;
}

K statecol(State e,K x,short t) {
 TORCH_CHECK(x->t==98,"expecting table with ",statekey(e)," column, given ",kname(x->t));
 J i=statefind(e,x);
 if(i<0) return nullptr;
 K v=kK(kK(x->k)[1])[i];
 TORCH_CHECK(t==nh || v->t==t, statekey(e)," column expected as ",kname(t),", given as ",kname(v->t));
 return v;
}
 
// --------------------------------------------
// convenience functions to return state value
// --------------------------------------------
J statedepth(K x,J j)     {return statelong(State::depth,true,x,j);}
J stategroup(K x,J j)     {return statelong(State::parmgroup,true,x,j);}
S statemodule(K x,J j)    {return statesym(State::module,true,x,j);}
S statename(K x,J j)      {return statesym(State::name,false,x,j);}
K stateoptions(K x,J j)   {return statedict(State::options,x,j);}
K stateparms(K x,J j)     {return statedict(State::parms,x,j);}
K statebuffers(K x,J j)   {return statedict(State::buffers,x,j);}
K statesize(K x,J j)      {return statelist(State::size,x,j);}

// ------------------------------------------------------
// nullsym - return null symbol or test for null symbol
// knull - return k null, i.e. (::)
// ------------------------------------------------------
S nullsym() {return env().nullsym;}
bool nullsym(S s) {return s==env().nullsym;}
bool nullsym(K x) {return x->t==-KS && nullsym(x->s);}
K knull() {K x=ka(101); x->g=0; return x;}

// --------------------------------------------------------------------------------------
// xnull  - true if null, i.e. (::)
// xempty - true if null or empty K list without type, i.e. :: or ()
// --------------------------------------------------------------------------------------
bool xnull(K x) {return x->t==101 && x->g==0;}
bool xnull(K x,J i) {return xind(x,i) && xnull(kK(x)[i]);}
bool xempty(K x) {return xnull(x) ? true : (x->t ? false : x->n==0);}
bool xempty(K x,J i) {return xind(x,i) && xempty(kK(x)[i]);}

// ---------------------------------------------------------------------------------------
// arraytype - true if k type is mapped to a tensor type
// xarray - true if x is an array that can be converted to a tensor, test up to m elements
// ---------------------------------------------------------------------------------------
static bool arraytype(short t) {
 if(t<0) t = -t;
 for(const auto &m:env().ktype)
  if(t==std::get<0>(m))
   return true;
 return false;
}

static bool xarray(K x,size_t d,Ksize& s,Ktype& t) {
 if(ptrtype(x)) {         // no pointers allowed
  return false;
 } else if(t==-128) {     // if no base type yet
  if(x->t) {              // if type, see if possible tensor type
   if(!arraytype(x->t)) return false;
   t=x->t;                     // 1st base type encountered
   if(t>0) s.push_back(x->n);  // unless scalar, track size
   return true;                // possible array
  } else {
   s.push_back(x->n);         // no base type yet, track size
   return x->n ? xarray(kK(x)[0],d+1,s,t) : (t=0,true);
  }
 } else if(x->t) {        // already defined base type
  if(t != x->t || t<0)    // type mismatch or subsequent scalar
   return false;          // not valid array
  else                    // also invalid if not matching size at depth
   return d==s.size()-1 && x->n==s.at(d);
 } else {
  if(d<s.size() && x->n == s[d]) // size at depth matches
   return x->n ? xarray(kK(x)[0],d+1,s,t) : true;
  else
   return false;
 }
}

bool xarray(K x,J m) {
 Ksize s; Ktype t=-128;
 if(x->t) {
  return xarray(x,0,s,t);
 } else if(ptrtype(x)) {
  return false;
 } else {
  if(x->n < m) m=x->n;
  for(J i=0; i<m; ++i)
   if(!xarray(kK(x)[i],0,s,t))
    return false;
  return true;
 }
}

// --------------------------------------------------------------------------------------
// xsym - if arg is k symbol, return true and set sym, else false
// xsyms - if sym scalar or non-empty sym list, set 1st sym and return true
//       - alternate form using symbol array ref
// --------------------------------------------------------------------------------------
bool xsym(K x) {return x->t==-KS;}
bool xsym(K x,J i) {return xind(x,i,KS) || (xind(x,i) && xsym(kK(x)[i]));}
bool xsym(K x,S &s) {return (x->t==-KS) ? s=x->s,true : false;}

bool xsym(K x,J i,S &s) {
 if(xind(x,i,KS))
  return s=kS(x)[i], true;
 else
  return xind(x,i) && xsym(kK(x)[i],s);
}

bool xsyms(K x,S &s) {
 if(xsym(x,s)) return true;
 else if(x->t == KS && x->n) return s=kS(x)[0],true;
 else return false;
}

bool xsyms(K x,SymArrayRef& s) {
 if(x->t==-KS) 
  return s=SymArrayRef(&x->s,1), true;
 else if(x->t==KS)
  return s=SymArrayRef(kS(x),x->n), true;
 else
  return false;
}

bool xsyms(K x,J i,SymArrayRef& s) {
 return xind(x,i) ? xsyms(kK(x)[i],s) : false;
}

// --------------------------------------------------------------------------------------
// xdev  - check sym for map to list of known devices, `cpu`cuda`cuda:0`cuda:1..
// xint64 - check for long scalar/list element and convert to int64_t or optional int64_t
// xlong - check for long scalar/list, set value(s) and return true else false
// xdouble - check for scalar double from k, set value and return true, false else
// xdict - return true if k value is a dictionary
// xstate - check for dictionary/table defining module state
// xsize - check for long(s)/double(s), set array ref/expanding array used for sizing
// --------------------------------------------------------------------------------------
bool xdev(K x,Device &d) {
 if(x->t==-KS) {
  for(const auto& m:env().device)
   if(x->s==std::get<0>(m)) return d=std::get<1>(m),true;
 }
 return false;
}

bool xdev(K x,J i,Device &d) {return xind(x,i) && xdev(kK(x)[i],d);}

bool xint64(K x,int64_t &j) {return (x->t == -KJ) ? j=x->j,true : false;}  //J -> int64_t (linux differentiates)
bool xint64(K x,J i,int64_t &j) {
 if(xind(x,i))                // i'th element of general list exists
  return xint64(kK(x)[i],j);  // check if long scalar, set & return true
 else if(xind(x,i,KJ))        // check for long list and valid index i
  return j=kJ(x)[i],true;
 else
  return false;
}
//bool xint64(K x,J i,int64_t &j) {return xind(x,i) && xint64(kK(x)[i],j);}

bool xint64(K x,c10::optional<int64_t> &j) {
 int64_t n; j=c10::nullopt;
 if(xint64(x,n)) {
  if(n != nj) j=n;
  return true;
 } else {
  return false;
 }
}

bool xint64(K x,J i,c10::optional<int64_t> &j) {
 int64_t n; j=c10::nullopt;
 if(xint64(x,i,n)) {
  if(n != nj) j=n;
  return true;
 } else {
  return false;
 }
}

bool xlong(K x,J &j) {return (x->t == -KJ) ? j=x->j,true : false;}       //check k scalar
bool xlong(K x,J i,J &j) {return xind(x,i) && xlong(kK(x)[i],j);}        //check k list element

bool xlong(K x,J &n,J *&v){                                        //check for k list of longs
 if(x->t == KJ){          n=x->n; v=kJ(x); return true;            //list of long ints
 } else if(x->t == -KJ){  n=1;    v=&x->j; return true;            //scalar long ok too
 } else if(x->t == 0 && x->n == 0) { n=0;  return true;            //empty,no type also ok
 } else { return false;
 }
}

bool xlong(K x,J i,J &n, J *&v) {return xind(x,i) && xlong(kK(x)[i],n,v);}  // check element of k list

bool xdouble(K x,double &f) {return (x->t == -KF) ? f=x->f,true : false;}    //check k scalar
bool xdouble(K x,J i,double &f) {return xind(x,i) && xdouble(kK(x)[i],f);}   //check k list element

bool xdouble(K x,J &n,double *&v){                                 //check for k list of doubles
 if(x->t == KF){          n=x->n; v=kF(x); return true;            //list of doubles
 } else if(x->t == -KF){  n=1;    v=&x->f; return true;            //scalar double ok too
 } else if(x->t == 0 && x->n == 0) { n=0;  return true;            //empty,no type also ok
 } else { return false;
 }
}
bool xdouble(K x,J i,J &n,double *&v) {return xind(x,i) && xdouble(kK(x)[i],n,v);}  // check element of k list

bool xdict(K x) {return x->t==99 && (kK(x)[0]->t==KS || (kK(x)[0]->t==0 && kK(x)[0]->n==0));}
bool xdict(K x,J i) {return xind(x,i) && xdict(kK(x)[i]);}

bool xstate(K x) {return xdict(x) || x->t==98;}
bool xstate(K x,J i) {return xind(x,i) && xstate(kK(x)[i]);}

// retrieve long integers from x -> IntArrayRef (linux clang/gcc require int64_t* from J*)
bool xsize(K x,IntArrayRef &s) {J n,*v; return (xlong(x,n,v)) ? s=IntArrayRef((int64_t*)v,n),true : false;}
bool xsize(K x,J i,IntArrayRef &s) {return xind(x,i) && xsize(kK(x)[i],s);}  //check element of k list

// retrieve long integers/doubles from x -> ExpandingArray ptr of longs/floats
bool xsize(K x,J d,int64_t *a) {
 bool b=false;
 if((b=x->t == -KJ)) {
   for(J i=0;i<d;++i) a[i]=x->j;
 } else if(x->t == KJ) {
  if((b=d == x->n))
   for(J i=0;i<d;++i) a[i]=kJ(x)[i];
  else
   TORCH_ERROR(d,"-element list of long integers expected, ",x->n," supplied");
 }
 return b;
}

bool xsize(K x,J d,double *a) {
 bool b=false; 
 if((b=x->t == -KF)) {
  for(J i=0;i<d;++i) a[i]=x->f;
 } else if(x->t == KF) {
  if((b=d == x->n))
   for(J i=0;i<d;++i) a[i]=kF(x)[i];
  else
   TORCH_ERROR(d,"-element list of doubles expected, ",x->n," supplied");
 }
 return b;
}

bool xsize(K x,J i,J d,int64_t *a) {return xind(x,i) && xsize(kK(x)[i],d,a);}
bool xsize(K x,J i,J d,double  *a) {return xind(x,i) && xsize(kK(x)[i],d,a);}

// ----------------------------------------------------------------------------------------------
// xten - check arg(s) for allocated ptr to tensor: set tensor & return true if found, else false
//      - 2nd form, return tensor pointer if found from k value, else null
// xout - check for tensor at end of list of args, return output tensor pointer
// xvec - check arg(s) for allocated vector of tensors
// xtensordict - check arg(s) for allocated dictionary of tensors
// ----------------------------------------------------------------------------------------------
bool xten(K x,Tensor &t) {
 if(auto* a=xtag(x))
  if(a->a==Class::tensor && a->c==Cast::tensor)
   return t=((Kten*)a)->t, true;
 return false;
}

Tensor* xten(K x) {
 if(auto* a=xtag(x))
  if(a->a==Class::tensor && a->c==Cast::tensor)
   return &((Kten*)a)->t;
 return nullptr;
}

bool xten(K x,J i,Tensor& t) {return xind(x,i) && xten(kK(x)[i],t);}
Tensor* xten(K x,J i) {return xind(x,i) ? xten(kK(x)[i]) : nullptr;}
Tensor* xout(K x) {return (!x->t && x->n>1) ? xten(x,x->n-1) : nullptr;}

TensorVector* xvec(K x) {
 if(auto* a=xtag(x))
  if(a->a==Class::vector && a->c==Cast::tensor)
   return &((Kvec*)a)->v;
 return nullptr;
}

TensorVector* xvec(K x,J i) {return xind(x,i) ? xvec(kK(x)[i]) : nullptr;}

TensorDict* xtensordict(K x) {
 if(auto* a=xtag(x))
  if(a->a==Class::dict && (a->c==Cast::tensor || a->c==Cast::parameter || a->c==Cast::buffer))
   return &((Kdict*)a)->d;
 return nullptr;
}

TensorDict* xtensordict(K x,J i) {return xind(x,i) ? xtensordict(kK(x)[i]) : nullptr;}

// ----------------------------------------------------------------------------------------------
// xtenarg - check arg(s) for a list of allocated tensors, or list of input arrays or mix of both
// ----------------------------------------------------------------------------------------------
bool xtenarg(K x,J i,Tensor& a,Tensor& b) {
 if(xten(x,i,a)) {
  if(!xten(x,i+1,b))
   b=kput(x,i+1).to(a.device());              // tensor b from array, move to a's device
  return true;
 } else if(xten(x,i+1,b)) {
  return a=kput(x,i).to(b.device()), true;    // tensor a from array, move to b's device
 } else {
  return a=kput(x,i), b=kput(x,i+1), false;   // both a & b from arrays, leave on cpu
 }
}

bool xtenarg(K x,J i,Tensor& a,Tensor& b,Tensor &c) {
 if(xten(x,i,a)) {
  if(!xten(x,i+1,b))
   b=kput(x,i+1).to(a.device());     // tensor b from array, move to a's device
  if(!xten(x,i+2,c))
   c=kput(x,i+2).to(a.device());     // tensor c from array, move to a's device
  return true;
 } else if(xten(x,i+1,b)) {          // tensor b supplied
  a=kput(x,i).to(b.device());        // tensor a from array, move to b's device
  if(!xten(x,i+2,c))
   c=kput(x,i+2).to(b.device());     // tensor c from array, move to b's device
  return true;
 } else if(xten(x,i+2,c)) {          // tensor c supplied
  a=kput(x,i).to(c.device());        // tensor a from array, move to c's device
  b=kput(x,i+1).to(c.device());      // tensor b from array, move to c's device
  return true;
 } else {                            // a,b,c from arrays -> cpu tensors
  return a=kput(x,i), b=kput(x,i+1), c=kput(x,i+2), false;   
 }
}

bool xtenarg(K x,Tensor& a,Tensor &b)           {return xtenarg(x,0,a,b);}
bool xtenarg(K x,Tensor& a,Tensor &b,Tensor &c) {return xtenarg(x,0,a,b,c);}
 
// -----------------------------------------------------
//  xtensors - return vector given arrays/tensors/vector
// -----------------------------------------------------
TensorVector xtensors(K x,bool& p,const char* c) {
 p=false;
 if(auto *a=xten(x)) {        p=true; return {*a};
 } else if(auto *a=xvec(x)) { p=true; return *a;
 } else if(x->t) {            return {kput(x)};
 } else {
  TORCH_CHECK(!x->t, c,": unexpected arg, ",kname(x));
  TensorVector v; Device d(DeviceType::CPU);
  for(J i=0; i<x->n; ++i) {
   K y=kK(x)[i];
   if(auto *a=xten(y)) {
    if(p) {
     TORCH_CHECK(d==a->device(), c,": given tensor on device ",a->device()," but previous tensor(s) on ",d);
    } else {
     p=true; d=a->device();    // tensor given, set flag and keep track of 1st device encountered
     for(auto &t:v) t=t.to(d); // put all k-array inputs so far on given tensor's device
    }
    v.emplace_back(*a);
   } else {
    TORCH_CHECK(!ptrtype(y), c,": arg[",i,"] unexpected, given ",kname(y));
    v.emplace_back(kput(y).to(d));
   }
  }
  return v;
 }
}

// ----------------------------------------------------------------------------
// xmodule - check arg(s) for allocated module pointer
// xloss - check arg(s) for allocated loss module
// xoptim - check arg(s) for allocated optimizer pointer
// xmodel - check arg(s) for allocated model pointer (module, loss & optimizer)
// ----------------------------------------------------------------------------
Kmodule* xmodule(K x) {auto* g=xtag(x); return (g && g->a==Class::module) ? g->kmodule() : nullptr;}
Kmodule* xmodule(K x,J i) {return xind(x,i) ? xmodule(kK(x)[i]) : nullptr;}

Kmodule* xloss(K x) {auto* g=xtag(x); return (g && g->a==Class::loss) ? g->kmodule() : nullptr;}
Kmodule* xloss(K x,J i) {return xind(x,i) ? xloss(kK(x)[i]) : nullptr;}

Kopt* xoptim(K x) {auto* g=xtag(x); return (g && g->a==Class::optimizer) ? (Kopt*)g : nullptr;}
Kopt* xoptim(K x,J i) {return xind(x,i) ? xoptim(kK(x)[i]) : nullptr;}

Kmodel* xmodel(K x) {auto* g=xtag(x); return (g && g->a==Class::model) ? (Kmodel*)g : nullptr;}
Kmodel* xmodel(K x,J i) {return xind(x,i) ? xmodel(kK(x)[i]) : nullptr;}

// ----------------------------------------------------------------------------
// xparm - check for model/module & parm/buffer name
// ----------------------------------------------------------------------------
bool xparm(K x,Cast c,S& s,Tensor& t) {
 bool b=false; s=nullptr;
 if(xsym(x,1,s)) {
  auto *g=xtag(x,0); const Tensor *a=nullptr;
  switch(g ? g->a : Class::undefined) {
   case Class::dict:
    a=g->dict().find(s);
    TORCH_CHECK(a, "unable to find dictionary ",tensortype(g->c)," `",s);
    break;
   case Class::loss:
   case Class::module:
   case Class::model:
   case Class::optimizer:
    a=findtensor(g->module(),s,c);
    TORCH_CHECK(a, "unable to find ",mapclass(g->a)," ",tensortype(c)," `",s);
    break;
   default:
    TORCH_ERROR("given ",tensortype(c)," name `",s,", expecting 1st arg of model, module or tensor dictionary, not ",kname(x,0));
    break;
  }
  b=true; t=*a;
 }
 return b;
}

// ------------------------------------------------------------------------------------------------------
// xnum - check for double or long int k scalar, set double & return true, else false
// xnum - check for number(float,double,long,int,short), set torch scalar & return true, else false
// xnumn - similar to xnum, but with optional scalars which remain unset if null scalar from k
// xnumt - similar to xnum, but also attempts to convert tensor to scalar
// xnumlist - take single value from k numeric list -> torch scalar
// xbyte - convert k bool,char,byte -> torch scalar
// xscalar - convert k number or byte -> torch scalar
// ------------------------------------------------------------------------------------------------------
bool xnum(K x,double &f) {
 switch(x->t) {
  case -KF: return f=x->f,true;
  case -KJ: return f=x->j,true;
  default: return false;
 }
}
bool xnum(K x,J i,double &f) {
 if(xind(x,i,KF))
  return f=kF(x)[i],true;
 else if(xind(x,i,KJ))
  return f=kJ(x)[i],true;
 else
  return xind(x,i) && xnum(kK(x)[i],f);
}

bool xnum(K x,Scalar& s) {
 switch(x->t) {
  case -KF:     return s=x->f, true;
  case -KE:     return s=x->e, true;
  case -KJ:     return s=(int64_t)x->j, true;
  case -KI:     return s=x->i, true;
  case -KSHORT: return s=x->h, true;
  default:      return false;
 }
}
bool xnum(K x,J i,Scalar& s) {return xind(x,i) && xnum(kK(x)[i],s);}

bool xnumn(K x,c10::optional<Scalar>& s) {
 switch(x->t) {
  case -KF:     if(x->f==x->f) s=x->f; return true;
  case -KE:     if(x->e==x->e) s=x->e; return true;
  case -KJ:     if(x->j!=nj) s=(int64_t)x->j; return true;
  case -KI:     if(x->i!=ni) s=x->i; return true;
  case -KSHORT: if(x->h!=nh) s=x->h; return true;
  default:      return false;
 }
}
bool xnumn(K x,J i,c10::optional<Scalar>& s) {return xind(x,i) && xnumn(kK(x)[i],s);}

bool xnumt(K x,Scalar& s) {
 Tensor t;
 if(xnum(x,s))      return true;
 else if(xten(x,t)) return s=t.item(), true;
 else               return false;
}

bool xnumt(K x,J i,Scalar& s) {return xind(x,i) && xnumt(kK(x)[i],s);}

bool xnumlist(K x,J i,Scalar &a) {
 switch(x->t) {
  case KF:     return a=kF(x)[i], true;
  case KE:     return a=kE(x)[i], true;
  case KJ:     return a=(int64_t)kJ(x)[i], true;
  case KI:     return a=kI(x)[i], true;
  case KSHORT: return a=kH(x)[i], true;
  case KB:
  case KC:     return a=kG(x)[i], true;
  default:     return false;
 }
}

bool xbyte(K x,Scalar &s) { return (x->t==-KB || x->t==-KC || xt==-KG) ? s=x->g,true : false;}
bool xbyte(K x,J i,Scalar &s) {return xind(x,i) && xbyte(kK(x)[i],s);}

bool xscalar(K x,Scalar &s) { return xnum(x,s) || xbyte(x,s);}
bool xscalar(K x,J i,Scalar &s) {return xind(x,i) && xscalar(kK(x)[i],s);}

// ------------------------------------------------------------------------------------------------------
// xbool - if value is boolean, set value and return true, else false
// mtype - match sym to/from TypeMeta(newer datatype from Caffe2)
// stype = match sym to/from TypeMeta to Dtype (torch::Dtype, aka at::ScalarType)
// xtype - symbol to scalar type or type meta, return true if scalar type/type meta set, else false
// xopt - sym(s) -> tensor options, return true if ok, false if not sym(s) or error if unknown sym
// xmode - check if sym, if matches a known tensor creation mode, set mode and return true else false
// xbacksym - check if sym, if matches back prop graph setting, set retain/create graph flags else false
// ------------------------------------------------------------------------------------------------------
bool xbool(K x,bool &b) {return (x->t == -KB) ? b=x->g,true : false;}
bool xbool(K x,J i,bool &b) {return xind(x,i) && xbool(kK(x)[i],b);}

TypeMeta mtype(S s) {
  for(const auto& m:env().dtype) if(s==std::get<0>(m)) return std::get<1>(m);
  TORCH_ERROR("unrecognized data type: ",(s==nullsym() ? "(null)" : s));
}

S mtype(TypeMeta t) {
  for(const auto& m:env().dtype) if(t==std::get<1>(m)) return std::get<0>(m);
  TORCH_ERROR("unrecognized data type: ",t);
}

Dtype stype(S s) {return torch::typeMetaToScalarType(mtype(s));}
S stype(Dtype t) {return mtype(torch::scalarTypeToTypeMeta(t));}
S stype(c10::optional<Dtype> t) {return mtype(torch::scalarTypeToTypeMeta(t ? *t : Dtype::Undefined));}

bool xtype(K x,TypeMeta &t) {if(x->t == -KS) return t=mtype(x->s), true; return false;}
bool xtype(K x,Dtype &t)    {if(x->t == -KS) return t=stype(x->s), true; return false;}
bool xtype(K x,c10::optional<Dtype> &t) {
 if(x->t != -KS)
  return false;
 if(nullsym(x->s))
   t=c10::nullopt;
  else
   t=stype(x->s);
 return true; 
}

bool xtype(K x,J i,TypeMeta &t)             {return xind(x,i) && xtype(kK(x)[i],t);}
bool xtype(K x,J i,Dtype &t)                {return xind(x,i) && xtype(kK(x)[i],t);}
bool xtype(K x,J i,c10::optional<Dtype> &t) {return xind(x,i) && xtype(kK(x)[i],t);}

bool xopt(S s,TensorOptions &o) {
 auto &e=env();
 for(const auto& m:e.device)
  if(s == std::get<0>(m)) return o=o.device(std::get<1>(m)), true;
 for(const auto& m:e.dtype)
  if(s == std::get<0>(m)) return o=o.dtype(std::get<1>(m)), true;
 for(const auto& m:e.layout)
  if(s == std::get<0>(m)) return o=o.layout(std::get<1>(m)), true;
 for(const auto& m:e.gradient)
  if(s == std::get<0>(m)) return o=o.requires_grad(std::get<1>(m)), true;
 for(const auto& m:e.pin)
  if(s == std::get<0>(m)) return o=o.pinned_memory(std::get<1>(m)), true;
 for(const auto& m:e.memory)
  if(s == std::get<0>(m)) return o=o.memory_format(std::get<1>(m)), true;
 return false;
}

bool xopt(K x,TensorOptions &o) {
 if (x->t == -KS || x->t == KS) {
  bool a=x->t < 0; I i,n=a ? 1 : x->n;
  for(i=0; i<n; ++i) {
   S s=a ? x->s : kS(x)[i];
   if (!xopt(s,o))
    TORCH_ERROR("unrecognized tensor option: `", s);
  }
  return true;
 } else {
  return false;
 }
}

bool xopt(K x,J i,TensorOptions &o) { return !x->t && 0<x->n && i<x->n && xopt(kK(x)[i],o);}

bool xmode(K x,S &s,Tensormode &m) {
 if(x->t == -KS) {
  for(const auto& v:env().tensormode)
   if(x->s == std::get<0>(v)) return s=x->s,m=std::get<1>(v), true;
  TORCH_ERROR("unrecognized tensor creation mode: ",x->s);
 }
 return false;
}

bool xmode(K x,J i,S &s,Tensormode &m) {return xind(x,i) && xmode(kK(x)[i],s,m);}

S modesym(Tensormode &m) {
 for(const auto& a:env().tensormode)
  if(m == std::get<1>(a)) return std::get<0>(a);
 TORCH_ERROR("unrecognized tensor creation mode: ",(I)m);
}

bool xbacksym(K x,bool& a,bool& b) {
 if(x->t == -KS) {
  for(const auto& s:env().backsym)
   if(x->s == std::get<0>(s)) return a=std::get<1>(s),b=std::get<2>(s), true;
  TORCH_ERROR("unrecognized setting for backprop: ",x->s,", expecting one of: free,retain,create or createfree");
 }
 return false;
}

bool xbacksym(K x,J i,bool& a,bool& b) {return xind(x,i) && xbacksym(kK(x)[i],a,b);}

// ------------------------------------------------------------------------------------------
// xpairs - initialize a set of name-value pairs given as an argument from k
// xpair - check the next name-value pair, set sym,numeric,list or general value
// xargc - return count of args to process given arg(s), offset, pairs structure to initiate
// xnone - return true if, given arg list and offset, no meaningful arg supplied
// ------------------------------------------------------------------------------------------
bool xpairs(K x,Pairs& p) {   // initialize Pairs structure from k value
 p.a=0, p.i=0, p.n=0;      // sets a: 1-dict,2-pairs,3-list,4-syms
 if(x->t==99) {
  K y=kK(x)[0];
  if(y->t==KS || !(y->t || y->n))
   p.a=1, p.n=y->n;
  else
   TORCH_ERROR("unexpected name,value dictionary with ",kname(kK(x)[0]->t)," as keys: ",kstring(x));
 } else if(x->t==KS) {
  if(x->n%2==0)
   p.a=4, p.n=x->n/2;
  else
   TORCH_ERROR("uneven no. of symbols for name,value pairs: ", kstring(x));
 } else if(!x->t) {
  if(!x->n) {                      // empty list
   p.a=2, p.n=0;
  } else if(kK(x)[0]->t==-KS) {    // list of (sym1;val1;sym2;val2..)
   if(x->n%2==0)
    p.a=3, p.n=x->n/2;
   else
    TORCH_ERROR("uneven no. of elements for name,value pairs in list: ",kstring(x));
  } else {                         // assume list of pairs if symbol in first pair
   K y=kK(x)[0];
   if(y->n==2 && (y->t==KS || (!y->t && kK(y)[0]->t==-KS)))
    p.a=2, p.n=x->n;
  }
 }
 return p.a ? (p.x=x,true) : false;
}

bool xpairs(K x,J i,Pairs& p) {return xind(x,i) && xpairs(kK(x)[i],p);}

static void xpair(Pairs& p,K x,J i) {
 if(x->t<0) {
  switch(x->t) {
   case -KS:     p.s=x->s; p.t=-KS; break;
   case -KB:     p.b=x->g; p.t=-KB; break;
   case -KSHORT: p.j=x->h; p.t=-KJ; break;
   case -KI:     p.j=x->i; p.t=-KJ; break;
   case -KJ:     p.j=x->j; p.t=-KJ; break;
   case -KE:     p.e=x->e; p.t=-KE; break;
   case -KF:     p.f=x->f; p.t=-KF; break;
   default: TORCH_ERROR("name-value pairs not implemented for ",kname(x->t)," value"); break;
  }
 } else if (i>-1) {
  if(i>=x->n)
   TORCH_ERROR("name,value index[",i,"] invalid for ",kname(x->t)," with ",x->n," elements");
  switch(x->t) {
   case 0:      xpair(p,kK(x)[i],-1); break;
   case KS:     p.s=kS(x)[i]; p.t=-KS; break;
   case KB:     p.b=kG(x)[i]; p.t=-KB; break;
   case KSHORT: p.j=kH(x)[i]; p.t=-KJ; break;
   case KI:     p.j=kI(x)[i]; p.t=-KJ; break;
   case KJ:     p.j=kJ(x)[i]; p.t=-KJ; break;
   case KE:     p.e=kE(x)[i]; p.t=-KE; break;
   case KF:     p.f=kF(x)[i]; p.t=-KF; break;
   default: TORCH_ERROR("name-value pairs not implemented for ",kname(x->t)," value"); break;
  }
 } else {
  p.v=x; p.t=x->t;
 }
}

bool xpair(Pairs& p) {
 if(p.i<0 || p.i>=p.n) return false;
 I i=p.i; p.k=nullptr; p.v=nullptr; K y;
 switch(p.a) {   
  case 1:  //dictionary
   p.k=kS(kK(p.x)[0])[i]; xpair(p,kK(p.x)[1],i); break;
  case 2:  //list of name-value pairs
   y=kK(p.x)[i];
   if(xlen(y)!= 2) {
    TORCH_ERROR("name,value pair[",i,"] has ",xlen(y)," elements (expected 2)");
   } else if(y->t==KS) {
    p.k=kS(y)[0]; xpair(p,y,1);
   } else if(!y->t && kK(y)[0]->t==-KS) {
    p.k=kK(y)[0]->s; xpair(p,kK(y)[1],-1);
   } else {
    TORCH_ERROR("name,value pair[",i,"] has no name symbol");
   }
   break;
  case 3:  //list of name,value,name,value..
   i*=2; y=kK(p.x)[i];
   if(y->t==-KS) {
    p.k=y->s; xpair(p,kK(p.x)[i+1],-1);
   } else {
    TORCH_ERROR("unrecognized name for pair, element[",i,"], expected symbol, received: ",kname(y->t));
   }
   break;
  case 4:  // list of symbols
    i*=2; p.k=kS(p.x)[i]; xpair(p,p.x,i+1); break;
  default: TORCH_ERROR("unrecognized name-value argument"); break;
 }
 return p.i++, true;
}

J xargc(K x,J i,Pairs& p) { // x:arg(s), i:offset, -1 if not applicable, p:pairs to initiate
 if(!x) {
  return 0;
 } else if(xdict(x)) {
  return xpairs(x,p), 0;             // dictionary of options, no other args to process
 } else if(x->t<0 || x->t>97) {
  return i>1 ? 0 : (i<0 ? 1 : 1-i);  // scalar arg, or table or different type of dictionary
 } else if(!x->n) {
  return 0;                          // return 0 for any empty list
 } else if(!(-1<i && i<=x->n)) {
  TORCH_ERROR("invalid offset: ",i,", for ",kname(x->t)," of length ",x->n);
 } else {
  return x->n-i-(x->n>i ? xpairs(x,x->n-1,p) : false);  // subtract name-value pairs from regular args to process
 }
}

bool xnone(K x,J i) {Pairs p; return !(xargc(x,i,p) || p.n);}

// ------------------------------------------------------------------------------------------
// perr - signal error in type of value given w'name-value pair
// plen  - signal length mismatch of input list if non-negative length supplied
// psym - check for symbol value in name/value pair, return symbol, else error
// ptype - check for symbol value that matches defined data type, e.g. `long`float`double
// pempty - return true if empty array value supplied for current name-value pair
// pbool - check for integral scalar with value 1 or 0, return true/false
// plong - check for integral scalar, return long int
// pdouble - check if numeric scalar, return double
// pnum - check for long/double, set torch scalar
// psize - check if integral scalar or list, set IntArrayRef or ExpandingArray, else error
// pdoubles - check if long/double scalar or list, set DoubleArrayRef, else error
// pten - attempt to define a tensor from provided scalar or array
// ------------------------------------------------------------------------------------------
void perr(const Pairs& p,const char* s) {TORCH_ERROR("option: ",p.k," is a ",kname(p.t),", expected a ",s);}

static void plen(const Pairs& p,J n,J m) {
 if(n==0 && (p.t<0 || m)) {
   TORCH_ERROR("option: ",p.k," requires zero elements, but single scalar value supplied");
 } else if(n>0 && (p.t>=0 && m!=n)) {
  TORCH_ERROR("option: ",p.k," requires ",n," elements, but ",m," supplied");
 }
}

S psym(const Pairs& p) {if(p.t!=-KS) perr(p,"symbol"); return p.s;}
Dtype ptype(const Pairs& p) {if(p.t!=-KS) perr(p,"symbol"); return stype(p.s);}
bool pempty(const Pairs& p) {return p.t==101 ? true :  p.t>=0 && p.v && !xlen(p.v);}
bool pbool(const Pairs& p) {if(p.t!=-KB) perr(p,"boolean"); return p.b;}
J plong(const Pairs& p) {if(p.t!=-KJ) perr(p,"long integer"); return p.j;}

double pdouble(const Pairs& p) {
 if(!(p.t==-KJ || p.t==-KF)) perr(p,"double or integer scalar");
 return p.t==-KJ ? p.j : p.f;
}

void pnum(const Pairs& p,torch::Scalar &s) {
 switch(p.t){
  case -KJ: s=(int64_t)p.j; break;
  case -KF: s=p.f; break;
  default: perr(p,"number"); break;
 }
}

void psize(const Pairs& p,IntArrayRef &s,J n) {
 if(p.t==-KJ)
  s=IntArrayRef((int64_t*)&p.j,1);  // recast for linux clang/gcc to go from J* -> int64_t*
 else if(!(p.t==KJ && xsize(p.v,s)))
  perr(p,"a long integer scalar or list");
 plen(p,n,s.size());
}

void psize(const Pairs& p,J d,int64_t *a) {
 if(p.t == -KJ) {
   for(J i=0;i<d;++i) a[i]=p.j;
 } else if(p.t == KJ) {
  if(d == xlen(p.v))
   for(J i=0;i<d;++i) a[i]=kJ(p.v)[i];
  else
   plen(p,d,xlen(p.v));
 } else {
  perr(p,"long integer scalar or list");
 }
}

void psize(const Pairs& p,J d,double *a) {
 if(p.t == -KF) {
   for(J i=0;i<d;++i) a[i]=p.f;
 } else if(p.t == KF) {
  if(d == xlen(p.v))
   for(J i=0;i<d;++i) a[i]=kF(p.v)[i];
  else
   plen(p,d,xlen(p.v));
 } else {
  perr(p,"double precision scalar or list");
 }
}

void pdoubles(const Pairs& p,DoubleArrayRef &a,J n) {
 J j; F *f;
 if(p.t==-KF)
  a=DoubleArrayRef(&p.f,1);
 else if(p.t==KF && xdouble(p.v,j,f))
  a=DoubleArrayRef(f,j);
 else
  perr(p,"a double scalar or list");
 plen(p,n,a.size());
}

void pten(const Pairs& p,Tensor &t) {
 switch(p.t) {
  case 0: if(!(xten(p.v,t))) t=kput(p.v); break;
  case -KB: t=torch::full({},Scalar(p.b),maptype(KB)); break;
  case -KJ: t=torch::full({},Scalar((int64_t)p.j),maptype(KJ)); break;
  case -KE: t=torch::full({},Scalar(p.e),maptype(KE)); break;
  case -KF: t=torch::full({},Scalar(p.f),maptype(KF)); break;
  case KB:
  case KSHORT:
  case KI:
  case KJ:
  case KE:
  case KF: t=kput(p.v); break;
  default: perr(p,"numeric scalar/array or previously allocated tensor pointer");
 }
}

// ----------------------------------------------------------------------
// argname - translate enumeration to sym, e.g. Arg::tensor -> `tensor
// argtype - translate sym to enumeration, e.g. `tensor -> Arg::tensor
// arglist - translate vector of enumerations to symbol or list
// ----------------------------------------------------------------------
S argname(Arg a) {
 if(a==Arg::undefined)
  return nullsym();
 for(const auto& x:env().arg)
  if(std::get<1>(x)==a)
   return std::get<0>(x);
 TORCH_ERROR("unrecognized arg type: ",(I)a,", unable to map to name");
}

Arg argtype(S s,const char *c) {
 for(const auto& a:env().arg)
  if(std::get<0>(a)==s) return std::get<1>(a);
 TORCH_ERROR("unrecognized ",(c ? c : "input/result")," type: `",s);
}

K arglist(const Args& v) {
 if(v.size()==1)
  return ks(argname(v.front()));
 J i=0; K x=ktn(KS,v.size());
 for(const auto a:v)
  kS(x)[i++]=argname(a);
 return x;
}

// ----------------------------------------------------------------------
// kstring - return string representation of k value
// kout - output k value via "0N!"
// kcast - given data type and array, cast and return, i.e. 1h$x
// kbool - cast k value to boolean
// ----------------------------------------------------------------------
std::string kstring(K x) {
 std::string s;
 K a=k(0,(S)".Q.s1",r1(x),0);
 if(a && a->t==KC) {
  s.assign((S)kC(a),a->n);
  r0(a);
 }
 return s;
}

std::string kstring(K x,J i) {return kstring(xind(x,i) ? kK(x)[i] : x);}

KAPI ch(K x) {return k(0,(S)"enlist",r1(x),0);}
K kshow(K x) {return k(0,(S)"0N!",r1(x),0);}
K kcast(Ktype t,K x) {return k(0,(S)"$",kh(t),r1(x),0);}
K kbool(K x) {return kcast(1,x);}

// -----------------------------------------------------------------------------------------
// kfind - given list of symbols, find index of matching string, return -1 if not found
// klist - return k value from count and long/double pointer
// kex - true if given list is one unique value
// kexpand - given element count & data ptr from expanding array return scalar or list
// -----------------------------------------------------------------------------------------
J kfind(K k,const std::string &s) {
 if(k->t != KS) TORCH_ERROR("unable to look up `",s," in ",kname(k->t),", expecting symbols");
 for(J i=0; i<k->n; ++i) if(!s.compare(kS(k)[i])) return i;
 return -1;
}

K klist(J n,const int64_t *j) {K x=ktn(KJ,n); memcpy(kG(x),j,n*sizeof(int64_t)); return x;}
K klist(J n,const double  *f) {K x=ktn(KF,n); memcpy(kG(x),f,n*sizeof(double));  return x;}
K klist(J n,const c10::optional<int64_t> *j) {
 K x=ktn(KJ,n);
 for(J i=0;i<n;++i) kJ(x)[i] = j[i] ? j[i].value() : nj;
 return x;
}

static bool kex(J n,const c10::optional<int64_t> *e) {
 bool b=n>0;
 for(I i=1;i<n;++i) {
  if((e[i-1].has_value() != e[i].has_value()) ||
     (e[i-1].has_value() && e[i].has_value()  && e[i-1] != e[i]))
   return false;
 }
 return b;
}

template<typename T>static bool kex(J n,const T *e) {
 bool b=n>0; for(I i=1;i<n;++i) if(e[i-1]!=e[i]) return false; return b;
}

K kexpand(J n,const int64_t *e) {return kex<int64_t>(n,e) ? kj(e[0]) : klist(n,e);}
K kexpand(J n,const double  *e) {return kex<double> (n,e) ? kf(e[0]) : klist(n,e);}
K kexpand(J n,const c10::optional<int64_t> *e) {return kex(n,e) ? kj(e[0] ? e[0].value() : nj) : klist(n,e);}

// ---------------------------------------------------------------
// dvt - recursive fn to convert nested tree to (depth;value)
// dv - k api fn convert nested tree -> (depth;value) pairs
// ---------------------------------------------------------------
static K dvt(J d,K x,K z) {
 K y=x->t || !x->n ? x : kK(x)[0];
 if(y->t == -KS)                   
  y=ks(y->s);                      // create a new symbol from given scalar
 else if(y->t == KS && y->n == 1)  
  y=ks(kS(y)[0]);                  // create a new symbol from 1-element symbol vector
 else
  r1(y);                           // else, increment reference count for use directly
 z=jk(&z,knk(2, kj(d), y));        
 if(!x->t)                         // add depth,value pairs from nested list
  for(J i=1;i<x->n;i++)
   z=dvt(d+1,kK(x)[i],z);
 return z;
}

KAPI dv(K x) {
 KTRY
  return dvt(0,x,ktn(0,0));
 KCATCH("dv");
}

// ---------------------------------------------------------------------------------------------
// xdv - return 0 if not recognized as (depth;value) format, -1 for single pair, n-list of pairs
// dve - return x value not recognized as (depth;value)
// dvd - return depth for i'th depth-value arg
// dvv - return i'th value from depth-value scalar/list
// tdv - recursive fn to convert depth-value scalar/list to nested tree representation
// tree - k api function to take depth-value pairs and return nested tree representation
// ---------------------------------------------------------------------------------------------
J xdv(K x) {
 if(x->t) {
  return 0;
 } else if(x->n==2 && kK(x)[0]->t == -KJ) {
  return -1;
 } else {
  for(J i=0;i<x->n;++i)
   if(-1 != xdv(kK(x)[i])) return 0;
  return x->n;
 }
}

J xdv(K x,J i) {return xind(x,i) ? xdv(kK(x)[i]) : false;}

static K dve(K x) { 
 if(!x->t && x->n>0 && xarray(x,2)) {
  for(J i=0;i<x->n;++i)
   if(!xdv(kK(x)[i])) return kK(x)[i];
 }
 return x;
}

J dvd(K x,J i) {return kK(i<0 ? x : kK(x)[i])[0]->j;}
K dvv(K x,J i) {return kK(i<0 ? x : kK(x)[i])[1];}

static K tdv(K x,J i,J j,J n) {
 K v=dvv(x,i), z=ktn(i==j && v->t==-KS ? KS : 0,1);
 if(z->t) kS(z)[0]=v->s;
 else     kK(z)[0]=r1(v);
 J k=i+1;
 for(i=k;i<=j;++i) {
  if(i>k && dvd(x,i)-n==1) {
   z=jk(&z,tdv(x,k,i-1,n+1)); k=i;
  }
  if(i==j)
   z=jk(&z,tdv(x,k,j,n+1));
 }
 return z;
}

KAPI tree(K x) {
 KTRY
  J n=xdv(x);
  TORCH_CHECK(n, "unable to parse (value;depth): ", kstring(dve(x)));
  if(n==-1) return tdv(x, -1,     -1, 0);
  else      return tdv(x,  0, x->n-1, 0);
 KCATCH("tree");
}

// -----------------------------------------------------------------------------------------
// addref - add a new kptr to a shared tensor/module/optimizer, incrementing reference count
// kfree - free allocated object/vector of objects and remove from active pointer set
// xfree - call kfree if active pointer, else noop, returns true unless free attempt fails
// ptrlist - return true if k list/dictionary of active pointers to pytorch objects
// freelist - free list/dictionary of pointers (allows for possible duplicates)
// Free - k api fn to free a previously allocated object or all allocated objects
// -----------------------------------------------------------------------------------------
KAPI addref(K x) {
 KTRY
  auto *g=xtag(x);
  TORCH_CHECK(g, "addref not implemented for ",kname(x->t));
  switch(g->a) {
   case Class::tensor:    return kten(g->tensor());
   case Class::module:    return kmodule(g->c, g->moduleptr());
   case Class::loss:      return kmodule(g->c, g->moduleptr(), Class::loss);
   case Class::optimizer: return kopt   (g->c, g->optptr(), g->moduleptr());
   default: TORCH_ERROR("addref not implemented for ",mapclass(g->a));
  }
 KCATCH("addref");
}

static void kfree(Ktag *g,K x) {
 delete g, pointer().erase(kK(x)[0]->j);
}

bool kfree(K x) {
 if(auto *g=xtag(x))
   return kfree(g,x), true;
 else
   return false;
}

bool kfree(K x,J i) {return xind(x,i) ? kfree(kK(x)[i]) : false;}

bool xfree(K x) {return mapped(x) ? kfree(x) : true;}

void kfree(const std::vector<K>& v) {
 for(auto x:v) // vector may have duplicates, no error for stale pointers
  xfree(x);
}


static bool ptrlist(K x,K y=nullptr);
static bool ptrlist(K x,K y) {
 if(xdict(x))
  return ptrlist(kK(x)[1], kK(x)[0]);
 if(x->t) {
  return false;
 } else {
  for(J i=0; i<x->n; ++i) {
   K v=kK(x)[i];
   if(ptrtype(v)) {
    if(y)  { TORCH_CHECK(ptrflag(v), "free: stale pointer, key[`", kS(y)[i], "] of given dictionary");
    } else { TORCH_CHECK(ptrflag(v), "free: stale pointer, element[", i, "] of given list"); }
   } else {
    if(y)  { TORCH_ERROR("free: expecting pytorch object but dictionary key[`", kS(y)[i], "] is ",kname(v));
    } else { TORCH_ERROR("free: expecting pytorch object but element[",i,"] is ",kname(v)); }
   }
  }
  return true;
 }
}

static void freelist(K x) {
 if(xdict(x))
  x=kK(x)[1];
 for(J i=0;i<x->n;++i) 
  xfree(kK(x)[i]);
}

KAPI Free(K x){
 KTRY
  if(xempty(x)) {
   for(auto j:pointer()) delete (Ktag*)j;
   pointer().clear();
  } else if(auto *g=xtag(x)) {
    kfree(g,x);
  } else if(ptrlist(x)) {
    freelist(x);
  } else {
   TORCH_ERROR("free: expecting pytorch object or list/dictionary of objects, given ",kname(x));
  }
  return (K)0;
 KCATCH("free");
}

// -----------------------------------------------------------------------------------------
// Return - collect k value(s) from tensor/vector/dictionary, free, then return values to k
// use - reassign ptr to new tensor/vector/dictionary, free any source allocations
// -----------------------------------------------------------------------------------------
KAPI Return(K x) {
 KTRY
  K r; Ktag *g=xtag(x); auto a=g ? g->a : Class::undefined;
  switch(a) {
   case Class::tensor: r=kget(g->tensor()); break;
   case Class::vector: r=kget(g->vector()); break;
   case Class::dict:   r=kget(g->dict()); break;
   default: r=nullptr;
  }
  TORCH_CHECK(r, "return: expecting tensor, vector or dictionary, given ",kname(x));
  if(!kfree(x)) {
   r0(r);
   TORCH_ERROR("return: unable to free source ",mapclass(a));
  }
  return r;
 KCATCH("return");
}

KAPI use(K x,K y) {
 KTRY
  Ktag *g=xtag(x);
  TORCH_CHECK(g && (g->a==Class::tensor || g->a==Class::vector || g->a==Class::dict),
              "use: 1st arg must be tensor, vector or dictionary");
  if(x != y) {
   Ktag *h=xtag(y);
   TORCH_CHECK(!h || (h->a==Class::tensor || h->a==Class::vector || h->a==Class::dict),
              "use: unable to re-assign ",mapclass(g->a)," to given ",mapclass(h->a));
   switch(g->a) {
    case Class::tensor:
     TORCH_CHECK(!h || h->a==Class::tensor, "use: cannot re-assign tensor to ",mapclass(h->a));
     g->set(h ? h->tensor() : kput(y)); break;
    case Class::vector:
     g->set((h && h->a==Class::vector) ? h->vector() : vec(y,true)); break;
    case Class::dict:
     g->set((h && h->a==Class::dict) ? h->dict() : kputd(y)); break;
    default: TORCH_ERROR("use: not implemented for ",mapclass(g->a));
   }
   TORCH_CHECK(!h || !ptrflag(y) || kfree(y), "use: unable to free source ",mapclass(h->a));
  }
  return (K)0;
 KCATCH("use");
}

// -------------------------------------------------------------------------------
// firstdevice - find 1st device of tensor(s), if none defined, return null
// defaultdevice - return default device if given null optional device
// -------------------------------------------------------------------------------
c10::optional<Device> firstdevice(const Tensor& t) {
 return t.defined() ? t.device() : c10::optional<Device>(c10::nullopt);
}

c10::optional<Device> firstdevice(const TensorVector& v) {
 for(const auto& t:v)
  if(auto d=firstdevice(t))
   return d;
 return c10::optional<Device>(c10::nullopt);
}

c10::optional<Device> firstdevice(const TensorDict& a) {
 for(const auto& i:a.items())
  if(auto d=firstdevice(i.value()))
   return d;
 return c10::optional<Device>(c10::nullopt);
}

c10::optional<Device> firstdevice(const Input& x) {
 using Dev=c10::optional<Device>;
 return std::visit(
  c10::overloaded(
   [&](const auto& x)  -> Dev {return firstdevice(x);},
   [&](const Empty& x) -> Dev {return c10::nullopt;}
  ),x);
}

Device defaultdevice(const c10::optional<Device> d) {
 return d ? *d : torch::kCPU;
}

// ---------------------------------------------------------------------------------
// sync - syncronize CUDA device or device index
// ksync - api function to sync CUDA device by index, name, tensors, module or model
// ---------------------------------------------------------------------------------
void sync(int64_t i) {
  torch::cuda::synchronize(i);
}

void sync(const Device& d) {
 if(d.is_cuda())
  sync(d.index());
}

KAPI ksync(K x) {
 KTRY
  S s; int64_t i; 
  if(xempty(x)) {
   if(env().cuda)
    sync(c10::impl::getDeviceGuardImpl(Device(torch::kCUDA).type())->getDevice());
  } else if(xint64(x,i)) {
   sync(i);
  } else if (xsym(x,s)) {
   Device d(torch::kCUDA);
   if(xdev(x,d) && d.is_cuda()) {
    if(!d.has_index()) 
     d=c10::impl::getDeviceGuardImpl(d.type())->getDevice();
    sync(d);
   }
  } else {
   auto g=xtag(x); c10::optional<Device> d;
   TORCH_CHECK(g, "sync: unrecognized arg, expecting device, device index, tensor, module or model, given ",kname(x));
   switch(g->a) {
    case Class::tensor: d=firstdevice(g->tensor()); break;
    case Class::vector: d=firstdevice(g->vector()); break;
    case Class::dict:   d=firstdevice(g->dict()); break;
    case Class::loss:
    case Class::module:
    case Class::model:
     d=g->kmodule()->d ? g->kmodule()->d.value() : firstdevice(g->module().parameters());
     break;
    default: TORCH_ERROR("sync: not implemented for ",mapclass(g->a));
   }
   sync(defaultdevice(d));
  }
  return (K)0;
 KCATCH("sync");
}

// -----------------------------------------------------------------------------------------
// objdevice - return tensor device, or first device found if object w'multiple tensors
// objsize - size tensor, no. of tensors in vector, else count of parameters
// objnum - number of elements in tensor's underlying storage or sum across tensors
// objbytes - bytes allocated in tensor's storage or sum across tensors/parms/buffers
// kobj - k api fn returns table of ptr,object,device,dtype,size,number of elements
// -----------------------------------------------------------------------------------------
S objdevice(const Tensor& t) {return tensorsym(t,Attr::device);}
S objdevice(const TensorVector& v,S s) {return v.size() ? objdevice(v[0]) : s;}
S objdevice(const Module& m,S s) {return objdevice(m.parameters().size() ? m.parameters() : m.buffers(), s);}
S objdevice(const Optimizer& o,S s) {
 for(const auto& g:o.param_groups())
  if(g.params().size()) return objdevice(g.params(),s);
 return s;
}

static S objdevice(Ktag *x) {
 S s=nullsym();
 switch(x->a) {
  case Class::tensor:    return objdevice(x->tensor());
  case Class::vector:    return objdevice(x->vector(), s);
  case Class::dict:      return objdevice(x->dict().values(), s);
  case Class::optimizer: return objdevice(x->opt(), s);
  case Class::loss:      return objdevice(x->module().buffers(), s);
  case Class::module:
  case Class::model:     return objdevice(x->module().parameters(), s);
  default: return s;
 }
}

static J objsize(const Module& m,bool b) {
 return m.parameters().size() + (b ? m.buffers().size() : 0);
}

static J objsize(Cast c,const Optimizer& o,bool b) {
 return b ? buffersize(Attr::tensorcount, c, o) : osize(o);
}

static J objsize(Kmodel *m,bool b) {
 if(b) // count all tensors & buffers in module,loss & optimizer
  return objsize(m->module(),b) + objsize(m->kloss()->module(),b) + objsize(m->o.c, *m->o.o, b);
 else
  return objsize(m->module(),b);  // just count parameters in module
}
 
static K objsize(Ktag *x,bool b=false);  // if b, return tensor counts, else "size"
static K objsize(Ktag *x,bool b) {
 switch(x->a) {
  case Class::tensor:    return tensorsize(x->tensor(), Attr::size);
  case Class::vector:    return kj(x->vector().size());
  case Class::dict:      return kj(x->dict().size());
  case Class::module:
  case Class::loss:      return kj(objsize(x->module(),b));
  case Class::optimizer: return kj(objsize(x->c, x->opt(), b));
  case Class::model:     return kj(objsize((Kmodel*)x,b));
  default: return ktn(0,0);
 }
}

J objnum(int64_t x) {return 1;}
J objnum(double  x) {return 1;}
J objnum(const Tensor& t) {return t.defined() ? (t.is_sparse() ? objnum(t._values()) : t.storage().nbytes() / t.dtype().itemsize()) : 0;}
J objnum(const TensorVector& v) {J n=0; for(const auto& t:v) n+=objnum(t); return n;}
J objnum(const c10::optional<TensorVector>& v) {return v ? objnum(*v) : 0;}
J objnum(const TensorDeque&  v) {J n=0; for(const auto& t:v) n+=objnum(t); return n;}
J objnum(const Module& m) {return objnum(m.parameters()) + objnum(m.buffers());}
J objnum(Cast c,const Optimizer& o) {return buffersize(Attr::elements,c,o);}
static J objnum(Kmodel *x) {return objnum(x->module()) + objnum(x->kloss()->module()) + objnum(x->o.c,*x->o.o);}

static J objnum(Ktag *x) {
 switch(x->a) {
  case Class::tensor:    return objnum(x->tensor());
  case Class::vector:    return objnum(x->vector());
  case Class::dict:      return objnum(x->dict().values());
  case Class::module:
  case Class::loss:      return objnum(x->module());
  case Class::optimizer: return objnum(x->c, x->opt());
  case Class::model:     return objnum((Kmodel*)x);
  default: return nj;
 }
}

J objbytes(int64_t x) {return sizeof(int64_t);}
J objbytes(double  x) {return sizeof(double);}

// determine bytes allocated in tensor by looking at underlying storage
// if more than 1 reference to storage use element count of tensor * size of elements
// this is to avoid double counting cases like CUDA LSTM modules where a single storage is used for all parameters
// see: https://github.com/pytorch/pytorch/issues/57632
J objbytes(const Tensor& t) {
 if(t.defined()) {
  if(t.is_sparse())
   return objbytes(t._indices()) + objbytes(t._values());
  else if(t.storage().use_count()>1)
   return t.numel() * t.dtype().itemsize();
  else
   return t.storage().nbytes();
  } else {
   return 0;
  }
}

J objbytes(const TensorVector& v) {J n=0; for(const auto& t:v) n+=objbytes(t); return n;}
J objbytes(const c10::optional<TensorVector>& v) {return v ? objbytes(*v) : 0;}
J objbytes(const TensorDeque&  v) {J n=0; for(const auto& t:v) n+=objbytes(t); return n;}
J objbytes(const Module &m) {return objbytes(m.parameters()) + objbytes(m.buffers());}
J objbytes(Cast c,const Optimizer& o) {return buffersize(Attr::bytes,c,o);}
static J objbytes(Kmodel *x) {return objbytes(*x->q.m) + objbytes(*x->l.m) + objbytes(x->o.c,*x->o.o);}

static J objbytes(Ktag *x) {
 switch(x->a) {
  case Class::tensor:    return objbytes(x->tensor());
  case Class::vector:    return objbytes(x->vector());
  case Class::dict:      return objbytes(x->dict().values());
  case Class::module:
  case Class::loss:      return objbytes(x->module());
  case Class::optimizer: return objbytes(x->c, x->opt());
  case Class::model:     return objbytes((Kmodel*)x);
  default: return nj;
 }
}

KAPI kobj(K x) {
 KTRY
  TORCH_CHECK(xempty(x), "obj: empty arg expected");
  K k=ktn(KS,7),v=ktn(0,7); auto n=pointer().size(); size_t i=0;
  kS(k)[0]=cs("ptr");      kK(v)[0]=ktn(0,n);
  kS(k)[1]=cs("class");    kK(v)[1]=ktn(KS,n);
  kS(k)[2]=cs("device");   kK(v)[2]=ktn(KS,n);
  kS(k)[3]=cs("dtype");    kK(v)[3]=ktn(KS,n);
  kS(k)[4]=cs("size");     kK(v)[4]=ktn(0,n);
  kS(k)[5]=cs("elements"); kK(v)[5]=ktn(KJ,n);
  kS(k)[6]=cs("bytes");    kK(v)[6]=ktn(KJ,n);
  for(const auto j:pointer()) {
   auto *g=(Ktag*)j;
   kK(kK(v)[0])[i] = knk(1,kj(j));
   kS(kK(v)[1])[i] = mapclass(g->a);
   kS(kK(v)[2])[i] = objdevice(g);
   kS(kK(v)[3])[i] = g->a == Class::tensor ? tensorsym(g->tensor(), Attr::dtype) : nullsym();
   kK(kK(v)[4])[i] = objsize(g);
   kJ(kK(v)[5])[i] = objnum(g);
   kJ(kK(v)[6])[i] = objbytes(g);
   ++i;
  }
  return xT(xD(k,v));
 KCATCH("obj");
}

KAPI tensorcount(K x) {
 KTRY
  auto *g=xtag(x);
  TORCH_CHECK(g, "tensorcount: not implemented for kname(x)");
  switch(g->a) {
   case Class::tensor:    return kj(1);
   case Class::vector: 
   case Class::dict:
   case Class::module:
   case Class::loss:     
   case Class::optimizer:
   case Class::model:     return objsize(g,true);
   default: TORCH_ERROR("tensorcount: not implemented for ",mapclass(g->a));
  }
 KCATCH("tensorcount");
}

// -----------------------------------------------------------------------------------------
// kstate - retrieve module/loss/optimizer state: options, internal buffers & parameters
// to - convert tensor/module device and or data type, e.g. to[tensor;`cuda`float;0b]
// kdetail - return dictionary of attributes of given object and level of detail
// -----------------------------------------------------------------------------------------
KAPI kstate(K x) {
 KTRY
  Ktag *g=xtag(x); bool a=env().alloptions;
  if(!g) {
   g=xtag(x,0);
   TORCH_CHECK(g && x->n<=2, "state: module, loss, optimizer or model expected, with optional flag as 2nd arg");
   TORCH_CHECK(x->n<2 || xbool(x,1,a), "state: 2nd arg expected to be boolean flag for all options or just defaults, given ",kname(x,1));
  }
  switch(g->a) {
   case Class::module:    return moduleget(a,true,g->module());
   case Class::loss:      return lossget(a,true,g->c,g->module());
   case Class::optimizer: return optget(a,true,g->c,g->opt(),g->module());
   case Class::model:     return modelget(a,true,(Kmodel*)g);
   default: TORCH_ERROR("state not defined for ",mapclass(g->a));
  }
 KCATCH("state")
}

static K to(K x,bool b,const char *s) {
 KTRY
  Ktag *g=xtag(x,0); bool a=false; TensorOptions o;
  TORCH_CHECK(g, s, ": expects 1st arg of tensor, vector, dictionary or module, along with option(s) for device, data type");
  TORCH_CHECK(x->n==2 || (x->n==3 && xbool(x,2,a)), s, ": expects 2-3 args, (",mapclass(g->a),";options) or (",mapclass(g->a),";options;async-flag)");
  if(auto *t=xten(x,1)) {
   o=o.device(t->device()).dtype(t->dtype());  // copy device & data type from example tensor
  } else {
   TORCH_CHECK(xopt(x,1,o), s,": 2nd argument not recognizable as option(s), e.g. device, data type, etc.");
  }
  TORCH_CHECK(!b || g->a==Class::tensor, s, ": expects tensor as first arg, given ",mapclass(g->a));
  switch(g->a) {
   case Class::tensor: return to((Kten*)g,o,a,b);
   case Class::dict:   return to(g->dict(),o,a), (K)0;
   case Class::vector: return to(g->vector(),o,a), (K)0;
   case Class::module:
   case Class::loss:   return to(g->kmodule(),o,a), (K)0;
   default: TORCH_ERROR(s, ": not implemented for ",mapclass(g->a));
  }
 KCATCH(s);
}

KAPI     To(K x) {return to(x, false, "to");}
KAPI copyto(K x) {return to(x, true,  "copyto");}

static K kinfo(K x,bool b,const char* e) {
 KTRY
  auto* g=xtag(x);
  TORCH_CHECK(g, e," not implemented for ",kname(x->t));
  switch(g->a) {
   case Class::tensor:     return tensorinfo(g->tensor(),b);
   default: TORCH_ERROR(e," not implemented for ",mapclass(g->a));
  }
 KCATCH(e);
}

KAPI info1(K x) {return kinfo(x, false, "info");}
KAPI info2(K x) {return kinfo(x, true,  "detail");}

// ------------------------------------------------------------------------------------
// cudadevices - return number of CUDA devices enabled or available CUDA device symbols
// cudadevice - k interface to set/query current CUDA device, e.g. `cuda:0 
// defaultdevice - return `cuda if any cuda devices available, else `cpu
// ------------------------------------------------------------------------------------
KAPI cudadevices(K x) {
 if(xnull(x)) {
  return kj(env().cuda);
 } else if(xempty(x)) {
  K s=ktn(KS,0);
  for(const auto& m:env().device) if((std::get<1>(m)).is_cuda()) js(&s,std::get<0>(m));
  return s;
 } else {
  return KERR("cudadevices[] returns count of available GPUs, cudadevices() returns CUDA syms");
 }
}

KAPI cudadevice(K x) {
 KTRY
  TORCH_CHECK(env().cuda, "no CUDA device available");
  Device d(torch::kCUDA);
  auto *g = c10::impl::getDeviceGuardImpl(d.type());
  if(xempty(x)) {
   for(const auto &m:env().device)
    if(g->getDevice()==std::get<1>(m)) return ks(std::get<0>(m));
   TORCH_ERROR("unable to map CUDA device: ",g->getDevice().index()," to symbol");
  } else if(xdev(x,d) && d.is_cuda() && d.has_index()) {
   return g->setDevice(d), K(0);
  } else {
   return KERR("unrecognized CUDA device, expecting cuda with valid device number, e.g. `cuda:0");
  }
 KCATCH("unable to query/set CUDA device")
}

static K defaultdevice() {
 auto d=Device(env().cuda ? DeviceType::CUDA : (torch::globalContext().hasMPS() ? DeviceType::MPS : DeviceType::CPU));
 for(const auto& c:env().device)
  if(std::get<1>(c)==d) return ks(std::get<0>(c));
 return KERR("unable to get default device");
}

// ---------------------------------------------------------------------------------------------
// optdev - map given device to corresponding symbol, e.g. DeviceType::CPU -> `cpu
// optdtype - map given data type to corresponding symbol, e.g. torch::kLong -> `long
// optlayout - map layout to corresponding symbol, e.g. torch::kSparse -> `sparse
// optgrad - map requires gradient flag to `grad or `nograd
// optpin - map pinned memory true/false to `pinned or `unpinned
// optmemory - map memory format to symbol, e.g. torch::MemoryFormat::ChannelsLast -> `channel2d
// optmap - given tensor options, return dictionary of attribute -> setting
// optkey - symbol keys/cols for tensor option dictionary or table
// optval - symbol vector/lists of option values
// ---------------------------------------------------------------------------------------------
S& optdev(const Device& d) {
 for(auto& m:env().device) if(d==std::get<1>(m)) return std::get<0>(m);
 TORCH_ERROR("unrecognized device: ",d);
}

S& optdtype(const TypeMeta& t) {
 for(auto& m:env().dtype) if(t==std::get<1>(m)) return std::get<0>(m);
 TORCH_ERROR("unrecognized data type: ",t);
}

S& optdtype(ScalarType s) {return optdtype(torch::scalarTypeToTypeMeta(s));}

S& optlayout(const torch::Layout& l) {
 for(auto& m:env().layout) if(l==std::get<1>(m)) return std::get<0>(m);
 TORCH_ERROR("unrecognized layout: ",l);
}

S& optgrad(const bool& g) {
 for(auto& a:env().gradient) if(g==std::get<1>(a)) return std::get<0>(a);
 TORCH_ERROR("unrecognized gradient setting: ",g);
}

S& optpin(const bool& p) {
 for(auto& a:env().pin) if(p==std::get<1>(a)) return std::get<0>(a);
 TORCH_ERROR("unrecognized pinned memory setting");
}

S& optmemory(const c10::optional<torch::MemoryFormat>& m) {
 if(!m) return optmemory(torch::MemoryFormat::Contiguous);
 for(auto& a:env().memory) if(m==std::get<1>(a)) return std::get<0>(a);
 TORCH_ERROR("unrecognized memory format");
}

torch::MemoryFormat optmemory(S s) {
 for(auto& a:env().memory) if(s==std::get<0>(a)) return std::get<1>(a);
 TORCH_ERROR("unrecognized memory format: ",s);
}

K optkey() {
 K x=ktn(KS,6);
 kS(x)[0]=mapattr(Attr::device);
 kS(x)[1]=mapattr(Attr::dtype);
 kS(x)[2]=mapattr(Attr::layout);
 kS(x)[3]=mapattr(Attr::gradient);
 kS(x)[4]=mapattr(Attr::pinned);
 kS(x)[5]=mapattr(Attr::memory);
 return x;
}

K optval(K x,J i) {
 for(J j=0;j<x->n;++j)
  if(x->t==KS)
   kS(x)[j]=nullsym();
  else
   kS(kK(x)[j])[i]=nullsym();
 return x;
}

K optval(const TensorOptions &o,K x,J i) {
 if(x->t==KS) {
  kS(x)[0]=optdev(o.device());
  kS(x)[1]=optdtype(o.dtype());
  kS(x)[2]=optlayout(o.layout());
  kS(x)[3]=optgrad(o.requires_grad());
  kS(x)[4]=optpin(o.pinned_memory());
  kS(x)[5]=optmemory(o.memory_format_opt());
 } else {
  kS(kK(x)[0])[i]=optdev(o.device());
  kS(kK(x)[1])[i]=optdtype(o.dtype());
  kS(kK(x)[2])[i]=optlayout(o.layout());
  kS(kK(x)[3])[i]=optgrad(o.requires_grad());
  kS(kK(x)[4])[i]=optpin(o.pinned_memory());
  kS(kK(x)[5])[i]=optmemory(o.memory_format_opt());
 }
 return x;
}

K optval(const Tensor &t,K x,J i) {
 if(!t.defined()) {
  return optval(x,i);
 } else if(x->t==KS) {
  kS(x)[0]=optdev(t.device());
  kS(x)[1]=optdtype(t.dtype());
  kS(x)[2]=optlayout(t.layout());
  kS(x)[3]=optgrad(t.requires_grad());
  kS(x)[4]=optpin(t.is_sparse() ? false : t.is_pinned());
  kS(x)[5]=optmemory(t.suggest_memory_format());
 } else {
  kS(kK(x)[0])[i]=optdev(t.device());
  kS(kK(x)[1])[i]=optdtype(t.dtype());
  kS(kK(x)[2])[i]=optlayout(t.layout());
  kS(kK(x)[3])[i]=optgrad(t.requires_grad());
  kS(kK(x)[4])[i]=optpin(t.is_sparse() ? false : t.is_pinned());
  kS(kK(x)[5])[i]=optmemory(t.suggest_memory_format());
 }
 return x;
}

K optmap(const Tensor &t) {
 K k=optkey(); return xD(k,optval(t,ktn(KS,k->n)));
}

K optmap(const TensorOptions &o) {
 K k=optkey(); return xD(k,optval(o,ktn(KS,k->n)));
}

// ---------------------------------------------------------------------------------------------
// cset - map between k symbol and eunumeration for configuration settings
// getsetting - return k scalar for given configuration setting
// getsettings - return k dictionary for all settings
// setflag - set boolean configuration setttings, e.g. stackframe on/off
// setlong - set long integer values, threads & inter-op threads
// setting - main k interface to show or change configuration settings
// config - print or return strings of pytorch config (CUDA capability, build options, etc.)
// version - return string or float version, e.g. "1.8.1" or 1.0801
// ---------------------------------------------------------------------------------------------
static Setting cset(S s) {
 for(const auto& c:env().cset)
  if(s==std::get<0>(c)) return std::get<1>(c);
 TORCH_ERROR("unrecognized configuration setting: ",s);
}

static K getsetting(Setting s) {
 switch(s) {
  case Setting::mkl:                return kb(torch::hasMKL());
  case Setting::openmp:             return kb(torch::hasOpenMP());
  case Setting::threads:            return kj(torch::get_num_threads());
//case Setting::threads:            return kj(torch::hasOpenMP() ? torch::get_num_threads() : 1);
  case Setting::interopthreads:     return kj(torch::get_num_interop_threads());
//case Setting::interopthreads:     return kj(torch::hasOpenMP() ? torch::get_num_interop_threads() : 1);
  case Setting::mps:                return kb(torch::globalContext().hasMPS());
  case Setting::cuda:               return kb(torch::cuda::is_available());
  case Setting::magma:              return kb(torch::hasMAGMA());
  case Setting::cudnn:              return kb(torch::cuda::cudnn_is_available());
  case Setting::cudnnversion:       return kj(torch::cuda::cudnn_is_available() ? torch::globalContext().versionCuDNN() : nj);
  case Setting::cudadevices:        return kj(env().cuda);
  case Setting::benchmark:          return kb(torch::globalContext().benchmarkCuDNN());
  case Setting::deterministic:      return kj(torch::globalContext().deterministicAlgorithms() ? 
                                             (torch::globalContext().deterministicAlgorithmsWarnOnly() ? 1 : 2) : 0);
  case Setting::cudnndeterministic: return kb(torch::globalContext().deterministicCuDNN());
  case Setting::stackframe:         return kb(env().frame);
  case Setting::alloptions:         return kb(env().alloptions);
  case Setting::complexfirst:       return kb(env().complexfirst);
  default: TORCH_ERROR("unrecognized setting: ",(I)s);
 }
}

static K getsettings() {
 J i=0; const auto& c=env().cset;  K k=ktn(KS,c.size()),v=ktn(0,c.size());
 for(auto a:c) {
  kS(k)[i]=std::get<0>(a);
  kK(v)[i++]=getsetting(std::get<1>(a));
 }
 return xD(k,v);
}

static void setflag(S s,Setting c,bool b) {
 switch(c) {
  case Setting::benchmark:          torch::globalContext().setBenchmarkCuDNN(b); break;
  case Setting::cudnndeterministic: torch::globalContext().setDeterministicCuDNN(b); break;
  case Setting::stackframe:         env().frame=b; break;
  case Setting::alloptions:         env().alloptions=b; break;
  case Setting::complexfirst:       env().complexfirst=b; break;
  default: TORCH_ERROR("setting: cannot set flag for ",s); break;
 }
}

static void setlong(S s,Setting c,J n) {
 TORCH_CHECK(torch::hasOpenMP(), "cannot set threads, no OpenMP capability detected");
 switch(c) {
  case Setting::threads:        torch::set_num_threads(n); break;
  case Setting::interopthreads: torch::set_num_interop_threads(n); break;
  case Setting::deterministic:
   if(n==0) {
    torch::globalContext().setDeterministicAlgorithms(false,false);
   } else if(n==1) {
    torch::globalContext().setDeterministicAlgorithms(true,true);
   } else if(n==2) {
    torch::globalContext().setDeterministicAlgorithms(true,false);
   } else {
    TORCH_ERROR("deterministic: use 0-turn off, 1-warn only, 2-error if non-deterministic algorithm");
   }
   break;
  default: TORCH_ERROR("setting: cannot set value for ",s); break;
 }
}

KAPI setting(K x) {
 KTRY
  S s;
  if(xempty(x)) {
   return getsettings();
  } else if(xsym(x,s)) {
   return getsetting(cset(s));
  } else if(xsym(x,0,s) && x->n==2) {
   bool b; J n; Setting c=cset(s);
   if(xbool(x,1,b))
    setflag(s,c,b);
   else if(xlong(x,1,n))
    setlong(s,c,n);
   else
    TORCH_ERROR("unable to change setting: ",s," with value ",kstring(x,1));
   return (K)0;
  } else {
   TORCH_ERROR("setting: unrecognized arg(s), expecting empty arg or sym, (sym;bool) or (sym;long)");
  }
 KCATCH("setting");
}

KAPI config(K x) {
 KTRY
  auto c1=torch::show_config(),c2=torch::get_parallel_info();
  if(xnull(x)) {
   std::cerr << c1 << "\n";
   std::cerr << c2 << "\n";
   return (K)0;
  } else if(xempty(x)) {
   std::stringstream s1(c1),s2(c2); std::string t; K z=ktn(0,0);
   while(std::getline(s1,t,'\n')) jk(&z,kp((S)t.c_str()));
   while(std::getline(s2,t,'\n')) jk(&z,kp((S)t.c_str()));
   return z;
  } else {
   return KERR("config expects empty argument: config[] prints to stderr, config() returns strings");
  }
 KCATCH("config");
}

KAPI version(K x) {
 KTRY
  TORCH_CHECK(xempty(x), "version: unexpected arg(s), use version() for string, version[] for number");
  return xnull(x) ? kf(TORCH_VERSION_MAJOR + TORCH_VERSION_MINOR/100.0 + TORCH_VERSION_PATCH/10000.0)
                  : kp((S)C10_STRINGIZE(TORCH_VERSION_MAJOR) "." C10_STRINGIZE(TORCH_VERSION_MINOR) "." C10_STRINGIZE(TORCH_VERSION_PATCH));
 KCATCH("version");
}

// -----------------------------------------------------------------------------------
// deviceseed - query/set seed for given device, return initial seed in use for device
// seedmap - returns map of device sym -> seed
// kseed - k interface to query/set device seed or query/reset seed for all devices
// -----------------------------------------------------------------------------------
J deviceseed(Device &d, bool b=false,J s=0) { // d:device, b:set flag, s:seed to set
 auto g=torch::globalContext().defaultGenerator(d);
 if(b) {
  if(null(s))
   g.seed();
  else
   g.set_current_seed(s);
 }
 return g.current_seed();
}

static K seedmap() {
 auto a=env().device; auto n=a.size(); I i=0; K k=ktn(KS,n),v=ktn(KJ,n);
 for(auto& m:a)
  kS(k)[i]=std::get<0>(m),kJ(v)[i++]=deviceseed(std::get<1>(m));
 return xD(k,v);
}

KAPI kseed(K x) {
 KTRY
  Device d(DeviceType::CPU); J s;
  if(xempty(x)) {                 // if empty, report on seed for all devices
   return seedmap();
  } else if(xlong(x,s)) {         // set single random seed across all devices
   if(null(s)) s=c10::detail::getNonDeterministicRandom();
   torch::manual_seed(s);
   return (K)0;
  } else if(xdev(x,d)) {          // query initial random seed for given device
   return kj(deviceseed(d));
  } else if(xdev(x,0,d) && xlong(x,1,s) && x->n==2) {  // set seed for given device
   deviceseed(d,true,s);
   return (K)0;
  } else {
   return KERR("unrecognized arg(s) for seed, expected one of: device, seed or (device;seed)");
  }
 KCATCH("unable to set/retrieve random seed(s)");
}

// -----------------------------------------------------------------------------------------
// query object attributes, e.g. tensor/vector and other object attributes
// -----------------------------------------------------------------------------------------
// modulename - name of module for forward calculation input/output, null if undefined
// moduleattr - handle a limited set of module attributes
// modelattr - handle a limited set of model attributes
// attr - attempt to get attribute of given object (more attributes implemented for tensors)
// -----------------------------------------------------------------------------------------
static S modulename(Kmodule *m,bool b) {
 auto c=b ? m->f.in() : m->f.out(); // get module type for forward input/output
 return c==Cast::undefined ? nullsym() : (m->a==Class::loss ? lmap(c) : msym(c));
}

static K moduleattr(Kmodule *m,Ktype k,Attr a) {
 switch(a) {
  case Attr::ref:          return kj(m->moduleptr().use_count());
  case Attr::ptr:          return kj((intptr_t)m->moduleptr().get());
  case Attr::device:       return ks(objdevice(m->module(), optdev(Device(torch::kCPU))));
  case Attr::dtype:
  case Attr::ktype:        return dictattr(m->module().named_parameters(),k,a); 
  case Attr::size:         return kj(m->module().parameters().size());
  case Attr::bytes:        return kj(objbytes(m->module()));
  case Attr::elements:     return kj(objnum(m->module()));
  case Attr::gradflag:
  case Attr::gradient:     return dictattr(m->module().named_parameters(),k,a);
  case Attr::inputmodule:  return ks(modulename(m,true));
  case Attr::outputmodule: return ks(modulename(m,false));
  default: TORCH_ERROR(mapattr(a),": not implemented for ",(m->a==Class::loss ? "loss " : ""),"modules");
 }
}

static K optattr(Cast c,const Optptr& o,Ktype k,Attr a) {
 switch(a) {
  case Attr::ptr:      return kj((intptr_t)o.get());
  case Attr::ref:      return kj(o.use_count());
  case Attr::size:     return kj(osize(*o));
  case Attr::bytes:    return kj(objbytes(c,*o));
  case Attr::elements: return kj(objnum(c,*o));
  default: TORCH_ERROR(mapattr(a),": not implemented for optimizers");
 }
}

static K modelattr(Kmodel *m,Ktype k,Attr a) {
 switch(a) {
  case Attr::size:         return kj(m->module().parameters().size());
  case Attr::bytes:        return kj(objbytes(m));
  case Attr::elements:     return kj(objnum(m));
  case Attr::dtype:
  case Attr::ktype:
  case Attr::gradflag:
  case Attr::gradient: 
  case Attr::inputmodule:
  case Attr::outputmodule: return moduleattr(m->kmodule(), k, a);
  default: TORCH_ERROR(mapattr(a),": not implemented for ",mapclass(m->a));
 }
}

K attr(K x,Ktype k,Attr a) {
 KTRY
  auto* g=xtag(x);
  TORCH_CHECK(g, mapattr(a),": unrecognized arg(s) - ",kname(x->t));
  switch(g->a) {
   case Class::tensor:    return tensorattr(g->tensor(),k,a);
   case Class::vector:    return vectorattr(g->vector(),k,a);
   case Class::dict:      return dictattr(g->dict(),k,a);
   case Class::module:
   case Class::loss:      return moduleattr(g->kmodule(),k,a);
   case Class::optimizer: return optattr(g->c, g->optptr(), k, a);
   case Class::model:     return modelattr((Kmodel*)g, k, a);
   default: TORCH_ERROR(mapattr(a),": not implemented for ",mapclass(g->a));
  }
 KCATCH("attr");
}

KAPI          dim(K x) {return attr(x, -KJ, Attr::dim);}
KAPI     densedim(K x) {return attr(x, -KJ, Attr::densedim);}
KAPI    sparsedim(K x) {return attr(x, -KJ, Attr::sparsedim);}
KAPI          nnz(K x) {return attr(x, -KJ, Attr::nnz);}
KAPI        numel(K x) {return attr(x, -KJ, Attr::numel);}
KAPI     elements(K x) {return attr(x, -KJ, Attr::elements);}
KAPI     itemsize(K x) {return attr(x, -KJ, Attr::itemsize);}
KAPI        bytes(K x) {return attr(x, -KJ, Attr::bytes);}
KAPI       offset(K x) {return attr(x, -KJ, Attr::offset);}
KAPI          ref(K x) {return attr(x, -KJ, Attr::ref);}
KAPI         sref(K x) {return attr(x, -KJ, Attr::sref);}
KAPI      weakref(K x) {return attr(x, -KJ, Attr::weakref);}
KAPI          ptr(K x) {return attr(x, -KJ, Attr::ptr);}
KAPI         sptr(K x) {return attr(x, -KJ, Attr::sptr);}
KAPI  inputmodule(K x) {return attr(x, -KS, Attr::inputmodule);}
KAPI outputmodule(K x) {return attr(x, -KS, Attr::outputmodule);}

KAPI kclass(K x) {
 KTRY
  Ktag *g=xtag(x);
  TORCH_CHECK(g, "class: need allocated torch object, e.g. tensor, module, given ",kname(x));
  return ks(mapclass(g->a));
 KCATCH("class");
}

S tensortype(Cast c) {
 for(auto& a:env().tensortype) if(c==std::get<1>(a)) return std::get<0>(a);
  TORCH_ERROR("unrecognized tensor type: cannot translate enumeration ",(I)c," to symbol");
}

KAPI kmapped(K x) {
 KTRY
  return kb(mapped(x));
 KCATCH("mapped");
}

KAPI objtype(K x) {
 KTRY
  S s; Ktag *g=xtag(x);
  TORCH_CHECK(g, "objtype: requires an allocated torch object, e.g. module, loss or optimizer, given ",kname(x));
  switch(g->a) {
   case Class::tensor:
   case Class::vector:
   case Class::dict:      s=tensortype(g->c); break;
   case Class::module:    s=msym(g->c); break;
   case Class::loss:      s=lmap(g->c); break;
   case Class::optimizer: s=omap(g->c); break;
   default: TORCH_ERROR("objtype: not implemented for ",mapclass(g->a)); break;
  }
  return ks(s);
 KCATCH("objtype");
}

KAPI     device(K x) {return xempty(x) ? defaultdevice() : attr(x, -KS, Attr::device);}
KAPI     layout(K x) {return attr(x, -KS, Attr::layout);}
KAPI   gradient(K x) {return attr(x, -KS, Attr::gradient);}
KAPI     gradfn(K x) {return attr(x, -KS, Attr::gradfn);}
KAPI     memory(K x) {return attr(x, -KS, Attr::memory);}
KAPI     result(K x) {return attr(x, -KS, Attr::result);}

KAPI dtype(K x) {
 TypeMeta t;
 if(xempty(x))
  return ks(optdtype(torch::get_default_dtype()));
 else if(xtype(x,t))
  return torch::set_default_dtype(t), (K)0;
 else
  return attr(x, -KS, Attr::dtype);
}

KAPI    defined(K x) {return attr(x, -KB, Attr::defined);}
KAPI  coalesced(K x) {return attr(x, -KB, Attr::coalesced);}
KAPI      ktype(K x) {return attr(x, -KC, Attr::ktype);}
KAPI       leaf(K x) {return attr(x, -KB, Attr::leaf);}
KAPI     pinned(K x) {return attr(x, -KB, Attr::pinned);}
KAPI sparseflag(K x) {return attr(x, -KB, Attr::sparseflag);}

KAPI contiguous(K x) {
 KTRY
  S s;
  if(xtag(x,0) && xsym(x,1,s) && x->n==2) {
   switch(optmemory(s)) {
    case torch::MemoryFormat::Preserve:
    case torch::MemoryFormat::Contiguous:     return attr(kK(x)[0], -KB, Attr::contiguous);
    case torch::MemoryFormat::ChannelsLast:   return attr(kK(x)[0], -KB, Attr::contiguous2d);
    case torch::MemoryFormat::ChannelsLast3d: return attr(kK(x)[0], -KB, Attr::contiguous3d);
    default: TORCH_ERROR("contiguous: unrecognized memory format, ", s);
   }
  } else {
   return attr(x, -KB, Attr::contiguous);
  }
 KCATCH("contiguous");
}

KAPI       size(K x) {return attr(x,  KJ, Attr::size);}
KAPI     stride(K x) {return attr(x,  KJ, Attr::stride);}

// ---------------------------------------------------------------------------
// castsym - given sym, get class & cast, e.g. Class::module & Cast::relu
// vectoroptions - return table of tensor options for each tensor in vector
// dictoptions - return table of tensor options for each tensor in dictionary
// ---------------------------------------------------------------------------
void castsym(S s,Class& a,Cast& c) {
 for(const auto& i:env().modules)
  if(std::get<0>(i)==s) {
   a=Class::module; c=std::get<1>(i);
   return;
  }
 for(const auto& i:env().loss)
  if(std::get<0>(i)==s) {
   a=Class::loss; c=std::get<1>(i);
   return;
  }
 for(const auto& i:env().opt)
  if(std::get<0>(i)==s) {
   a=Class::optimizer; c=std::get<1>(i);
   return;
  }
 a=Class::undefined; c=Cast::undefined;
}

Cast castsym(S s) {
 Class a; Cast c; castsym(s,a,c); return c;
}

static K vectoroptions(const TensorVector& v) {
 K k=optkey(); K y=ktn(0,k->n);
 for(J i=0; i<k->n; ++i)
  kK(y)[i]=ktn(KS,v.size());
 for(size_t i=0; i<v.size(); ++i)
  optval(v[i],y,i);
 return xT(xD(k,y));
}

static K dictoptions(const TensorDict& d) {
 K c=optkey(); K k=ktn(KS,d.size()),y=ktn(0,c->n); J i;
 for(i=0; i<y->n; ++i) kK(y)[i]=ktn(KS,d.size());
 i=0;
 for(const auto& a:d) {
  kS(k)[i]=cs(a.key().c_str());
  optval(a.value(),y,i++);
 }
 return xD(k,xT(xD(c,y)));
}

// -------------------------------------------------------------------------
// modulekeys - return symbol keys for module help table/dictionary
// modulehelp - return a table of module definitions or single dict of attrs
// -------------------------------------------------------------------------
static K modulekeys() {
 K k=ktn(KS,7);
 kS(k)[0]=cs("module");
 kS(k)[1]=cs("pytorch");
 kS(k)[2]=cs("forward");
 kS(k)[3]=cs("result");
 kS(k)[4]=cs("n");
 kS(k)[5]=cs("args");
 kS(k)[6]=cs("options");
 return k;
}

static AttrRef moduleref(bool b) {
 return b ? AttrRef(env().modules) : AttrRef(env().loss);
}

static K modulehelp(bool b) {  // true to return table of modules, else table of loss modules
 auto e=moduleref(b);
 J i=0,m=e.size();
 K k=modulekeys();
 K s=ktn(KS,m);   // module sym
 K d=ktn(0,m);    // pytorch description
 K f=ktn(KB,m);   // has non-templatized forward flag
 K r=ktn(KS,m);   // result type
 K n=ktn(KJ,m);   // no. of required args
 K a=ktn(0,m);    // arg type(s)
 K o=ktn(0,m);    // module creation options
 for(const auto& z:e) {
  kS(s)[i]=std::get<0>(z);
  kK(d)[i]=kp((S)std::get<3>(z));
  kG(f)[i]=std::get<4>(z);
  kS(r)[i]=argname(std::get<5>(z));
  kJ(n)[i]=std::get<6>(z);
  kK(a)[i]=arglist(std::get<8>(z));
  kK(o)[i++]=b ? modexample(std::get<1>(z)) : lossexample(std::get<1>(z));
 }
 return xT(xD(k,knk(7,s,d,f,r,n,a,o)));
}

static K modulehelp(bool b,Cast c,K d) {
 for(const auto& z:moduleref(b)) {
  if(std::get<1>(z)==c) {
    K v=knk(7,
     ks(std::get<0>(z)),           // module symbol
     kp((S)std::get<3>(z)),        // pytorch description
     kb(std::get<4>(z)),           // has non-template forward()
     ks(argname(std::get<5>(z))),  // result type
     kj(std::get<6>(z)),           // no. of required args for forward()
     arglist(std::get<8>(z)),      // input arg type(s)
     d ? d : (b ? modexample(c) : lossexample(c))); // options or example if none given
    return xD(modulekeys(),v);
   }
  }
  TORCH_ERROR("help: unable to resolve module enumeration: ",(I)c);
}

K modulehelp(Kmodule *m) {
 bool a=env().alloptions;
 bool b=m->a == Class::module;
 return modulehelp(b, m->c, b ? moduleoptions(a,false,m->c,m->module())
                              : lossoptions(a,m->c,m->module()));
}

// ---------------------------------------------------------------------------
// backmode - table of backward calculation modes (for help)
// dtypes - map of pytorch data type to k type
// ktypes - map of k types to pytorch data type
// ---------------------------------------------------------------------------
static K backmode() {
K k=ktn(KS,4), v=ktn(0,4); auto n=env().backsym.size(); size_t i=0;
  kS(k)[0]=cs("mode");        kK(v)[0]=ktn(KS,n);
  kS(k)[1]=cs("retain");      kK(v)[1]=ktn(KB,n);
  kS(k)[2]=cs("create");      kK(v)[2]=ktn(KB,n);
  kS(k)[3]=cs("description"); kK(v)[3]=ktn(0,n);
  for(const auto& a:env().backsym) {
   kS(kK(v)[0])[i]=std::get<0>(a);
   kG(kK(v)[1])[i]=std::get<1>(a);
   kG(kK(v)[2])[i]=std::get<2>(a);
   kK(kK(v)[3])[i]=kp((S)std::get<3>(a));
   ++i;
  }
  return xT(xD(k,v));
}

static K dtypes() {
 auto n=env().dtype.size(); K k=ktn(KS,n), v=ktn(KC,n); size_t i=0;
 for(const auto& a:env().dtype)
  kS(k)[i]=std::get<0>(a), kC(v)[i]=std::get<3>(a), ++i;
 return xD(k,v);
}

static K ktypes() {
 auto n=env().ktype.size(); K k=ktn(KC,n), v=ktn(KS,n); size_t i=0;
 for(const auto& a:env().dtype)
  kC(k)[i]=std::get<3>(a), kS(v)[i]=std::get<0>(a), ++i;
 return xD(k,v);
}

// ---------------------------------------------------------------------------
// helpclass - look for class symbol, return class enumeration or 'undefined'
// helptopic - map to/from symbol and help topic enumeration
// helpsym - return list of symbols for overall class help
// khelp - k api function, accepts empty arg, symbol or object
//       - flag indicates full table or dictionary of options only
// ---------------------------------------------------------------------------
static Class helpclass(S s) {
 for(const auto& a:env().kclass)
  if(std::get<0>(a)==s) return std::get<1>(a);
 return Class::undefined;
}

static Help helptopic(S s) {
 for(const auto& a:env().helptopic)
  if(std::get<0>(a)==s) return std::get<1>(a);
 return Help::undefined;
}

static S helptopic(Help h) {
 for(const auto& a:env().helptopic)
  if(std::get<1>(a)==h) return std::get<0>(a);
 return nullsym();
}

static K helpsym(void) {
 K k=ktn(KS,0), v=ktn(0,0);
 js(&k, helptopic(Help::backward));  jk(&v, kp((S)"backward calculation modes"));
 js(&k, helptopic(Help::device));    jk(&v, kp((S)"PyTorch device mapped to random seed"));
 js(&k, helptopic(Help::dtype));     jk(&v, kp((S)"PyTorch tensor types mapped to k types"));
 js(&k, helptopic(Help::ktype));     jk(&v, kp((S)"k types mapped to PyTorch tensor types"));
 js(&k, mapclass(Class::loss));      jk(&v, kp((S)"loss types, module options & forward args"));
 js(&k, mapclass(Class::module));    jk(&v, kp((S)"module types, options & forward args"));
 js(&k, mapclass(Class::optimizer)); jk(&v, kp((S)"optimizer types & options"));
 js(&k, mapclass(Class::tensor));    jk(&v, kp((S)"tensor creation options"));
 return xD(k,v);
}

K khelp(K x,bool h,const char *e) {
 KTRY
  S s;
  if(xempty(x) || nullsym(x)) {
   return h ? helpsym() : optmap(TensorOptions());
  } else if(xsym(x,s)) {
   Class a=helpclass(s); Help t;
   if(a != Class::undefined) {
    switch(a) {
     case Class::tensor:
     case Class::vector:
     case Class::dict:      return optmap(TensorOptions());
     case Class::module:    return modulehelp(true);
     case Class::loss:      return modulehelp(false);
     case Class::optimizer: return optdefaults(Cast::undefined);
     default: TORCH_ERROR(e, ": not implemented for ",mapclass(a));
    }
   } else if((t=helptopic(s)) != Help::undefined) {
    switch(t) {
     case Help::backward: return backmode();
     case Help::device:   return seedmap();
     case Help::dtype:    return dtypes();
     case Help::ktype:    return ktypes();
     default: TORCH_ERROR("help topic: ",s," not implemented");
    }
   } else {
    Cast c; castsym(s,a,c);
    TORCH_CHECK(c != Cast::undefined, e, ": help topic, `",s);
    switch(a) {
     case Class::module:    return h ? modulehelp(true,c,nullptr) : modexample(c);
     case Class::loss:      return h ? modulehelp(false,c,nullptr) : lossexample(c);
     case Class::optimizer: return optdefaults(c);
     default: TORCH_ERROR(e,": not implemented for ",mapclass(a));
    }
   }
  } else if(auto *g=xtag(x)) {
   bool a=env().alloptions;
   switch(g->a) {
    case Class::tensor:    return optmap(g->tensor());
    case Class::vector:    return vectoroptions(g->vector());
    case Class::dict:      return dictoptions(g->dict());
    case Class::module:    return h ? modulehelp(g->kmodule()) : moduleoptions(a,false,g->c,g->module());
    case Class::loss:      return h ? modulehelp(g->kmodule()) : lossoptions(a,g->c,g->module());
    case Class::optimizer: return optsettings(a,g->c,g->opt());
    default: TORCH_ERROR((h ? "help" : "options"),": not implemented for ",mapclass(g->a));
   }
  } else {
   TORCH_ERROR((h ? "help" : "options"),": unrecognized arg(s), expecting empty arg, name or object, given ",kname(x));
  }
 KCATCH(e);
}

KAPI    help(K x) {return khelp(x, true,  "help");}
KAPI options(K x) {return khelp(x, false, "options");}

// ---------------------------------------------------------------------------
// str - PyTorch string representation of tensor or module
// print_tensor - print first n elements of tensor (for module args)
// ---------------------------------------------------------------------------
KAPI str(K x) {
 KTRY
  Ktag *g=xtag(x); std::string s;
  TORCH_CHECK(g, "str: need allocated torch object, e.g. tensor, module, given ",kname(x));
  switch(g->a) {
   case Class::tensor:    s=c10::str(g->tensor()); break;
   case Class::loss:
   case Class::module:
   case Class::model:     s=c10::str(g->module()); break;
   default: TORCH_ERROR("str: not implemented for ",mapclass(g->a));
  }
  return kp(const_cast<S>(s.c_str()));
 KCATCH("class");
}

void print_tensor(std::ostream& s,int64_t n,const torch::Tensor& t) {
 if(t.dim()) {
  auto m=t.numel(); if(m<n) n=m;
  for(int64_t i=0; i<n; ++i)
   s << t.view(-1)[i].item<double>() << (i+1<n ? "," : (n<m ? ".." : ""));
 } else {
  s << t.item<double>();
 }
}

// ----------------------------------------------------------------------------------------------
// filewrite - write zero bytes if file writable, create any necessary dir(s), drop leading colon
//  pt
// png - write PNG file given name & array/tensor
// ----------------------------------------------------------------------------------------------
S filewrite(S s) {k(0,(S)"1:",ks(s),ktn(KB,0),0); return s[0]==':' ? s+1 : s;}

KAPI pt(K x) {
 KTRY
  S s; Tensor t;
  if(xsym(x,s)) {
   torch::load(t,s[0]==':' ? ++s : s);
   return kget(t);
  } else if (xsym(x,0,s) && xten(x,1,t)) {
   torch::save(t,s[0]==':' ? ++s : s);
  }
  return (K)0;
 KCATCH("pt");
}

KAPI png(K x) {
 KTRY
  S s; Tensor t;
  TORCH_CHECK(xsym(x,0,s) && x->n==2, "png: unrecognized arg(s), expects (PNG file name; image array/tensor)");
  if(!xten(x,1,t)) t=kput(x,1);
  auto a=t.to(torch::kCPU,torch::kByte).permute({1,2,0}).flatten(1);
  stbi_write_png(filewrite(s),t.size(2),t.size(1),t.size(0),a.data_ptr(),t.size(0)*t.size(2));
  return ks(s);
 KCATCH("png");
}

// -----------------------------------------------------------------------------------------
// initialize globals: device counts, device sym-int mapping, etc.
// kinit - called when shared library is first opened
// -----------------------------------------------------------------------------------------
Env&  env()  {static Env  e; return e;}
Esym& esym() {static Esym e; return e;}
std::unordered_set<J>& pointer() {static std::unordered_set<J> p; return p;}

static void kinit() __attribute__((constructor));

static void kinit() {
 C c[16]; auto &e=env(); auto &d=e.device;
 e.frame = false;                                       //no stack frame on error msg
 e.cuda = torch::cuda::device_count();                  //count of available CUDA devices
 d.emplace_back(cs("cpu"),Device(DeviceType::CPU));     //build map from sym->device
 if(e.cuda) {
  d.emplace_back(cs("cuda"),Device(DeviceType::CUDA));  //current CUDA device, `cuda
  for(I i=0; i<e.cuda; ++i) {                           //device 0-n, e.g. `cuda:0
   TORCH_CHECK(sizeof(c)>(size_t)snprintf(c,sizeof(c),"cuda:%d",i), "buffer too small for `cuda:",i,"`");
   d.emplace_back(ss(c),Device(DeviceType::CUDA,i));
  }
 }
 if(torch::globalContext().hasMPS()) {
  d.emplace_back(cs("mps"),Device(DeviceType::MPS));    //Mac Metal Performance Shading
  TORCH_CHECK(sizeof(c)>(size_t)snprintf(c,sizeof(c),"mps:%d",0), "buffer too small for `mps:0`");
  d.emplace_back(ss(c),Device(DeviceType::MPS,0));
 }
}

// -----------------------------------------------------------------------------------------
// fn - given dictionary, along with name, fn & arg count, adds function to dictionary
// fns - returns K dictionary with function names and code
// -----------------------------------------------------------------------------------------
void fn(K x,const char* s,void *f,I n){dictadd(x,s,dl(f,n));}

KAPI fns(K x){
 x=KDICT;
 fn(x, "dv",           KFN(dv),          1);
 fn(x, "tree",         KFN(tree),        1);
 fn(x, "addref",       KFN(addref),      1);
 fn(x, "free",         KFN(Free),        1);
 fn(x, "return",       KFN(Return),      1);
 fn(x, "use",          KFN(use),         2);
 fn(x, "obj",          KFN(kobj),        1);
 fn(x, "tensorcount",  KFN(tensorcount), 1);
 fn(x, "to",           KFN(To),          1);
 fn(x, "copyto",       KFN(copyto),      1);
 fn(x, "info",         KFN(info1),       1);
 fn(x, "detail",       KFN(info2),       1);
 fn(x, "state",        KFN(kstate),      1);
 fn(x, "setting",      KFN(setting),     1);
 fn(x, "config",       KFN(config),      1);
 fn(x, "version",      KFN(version),     1);
 fn(x, "cudadevice",   KFN(cudadevice),  1);
 fn(x, "cudadevices",  KFN(cudadevices), 1);
 fn(x, "sync",         KFN(ksync),       1);
 fn(x, "seed",         KFN(kseed),       1);
 fn(x, "png",          KFN(png),         1);
 fn(x, "dim",          KFN(dim),         1);
 fn(x, "densedim",     KFN(densedim),    1);
 fn(x, "sparsedim",    KFN(sparsedim),   1);
 fn(x, "nnz",          KFN(nnz),         1);
 fn(x, "numel",        KFN(numel),       1);
 fn(x, "elements",     KFN(elements),    1);
 fn(x, "itemsize",     KFN(itemsize),    1);
 fn(x, "bytes",        KFN(bytes),       1);
 fn(x, "offset",       KFN(offset),      1);
 fn(x, "ptr",          KFN(ptr),         1);
 fn(x, "sptr",         KFN(sptr),        1);
 fn(x, "ref",          KFN(ref),         1);
 fn(x, "sref",         KFN(sref),        1);
 fn(x, "result",       KFN(result),      1);
 fn(x, "weakref",      KFN(weakref),     1);
 fn(x, "class",        KFN(kclass),      1);
 fn(x, "mapped",       KFN(kmapped),     1);
 fn(x, "objtype",      KFN(objtype),     1);
 fn(x, "device",       KFN(device),      1);
 fn(x, "dtype",        KFN(dtype),       1);
 fn(x, "ktype",        KFN(ktype),       1);
 fn(x, "gradfn",       KFN(gradfn),      1);
 fn(x, "gradient",     KFN(gradient),    1);
 fn(x, "layout",       KFN(layout),      1);
 fn(x, "memory",       KFN(memory),      1);
 fn(x, "inputmodule",  KFN(inputmodule), 1);
 fn(x, "outputmodule", KFN(outputmodule),1);
 fn(x, "defined",      KFN(defined),     1);
 fn(x, "coalesced",    KFN(coalesced),   1);
 fn(x, "contiguous",   KFN(contiguous),  1);
 fn(x, "leaf",         KFN(leaf),        1);
 fn(x, "pinned",       KFN(pinned),      1);
 fn(x, "sparseflag",   KFN(sparseflag),  1);
 fn(x, "size",         KFN(size),        1);
 fn(x, "stride",       KFN(stride),      1);
 fn(x, "str",          KFN(str),         1);
 fn(x, "options",      KFN(options),     1);
 fn(x, "help",         KFN(help),        1);
 fn(x, "resolve",      KFN(Resolve),     1);

 tensorfn(x);
 mathfn(x);
 nnfn(x);
 lossfn(x);
 optfn(x);
 modelfn(x);
 return x;
}
