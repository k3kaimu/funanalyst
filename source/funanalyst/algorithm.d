module funanalyst.algorithm;

import std.complex;
import std.math;
import std.range;

import funanalyst.space;

LinearCombination!(X, basis)[] gramSchmidt(alias basis, X, alias innerProd)(size_t maxOrder)
{
    X[] initcoefs = [];
    alias F = LinearCombination!(X, basis);
    if(maxOrder == 0) return [];

    F[] funcs;


    foreach(order; 1 .. maxOrder + 1) {
        if(initcoefs.length != 0) initcoefs[$-1] = X(0);
        initcoefs ~= X(1);

        funcs ~= F(initcoefs);
    }

    return gramSchmidt!innerProd(funcs);
}


unittest
{
    alias F = Polynomial!double;

    static
    double uniformInnerProd(F f, F g)
    {
        double dst = 0;
        foreach(i; 0 .. f.coefs.length)
            foreach(j; 0 .. g.coefs.length)
                dst += f.coefs[i] * g.coefs[j] / (i + j + 1);

        return dst;
    }

    import std.algorithm, std.range;
    auto onbs = gramSchmidt!(F.basis, F.ParamType, uniformInnerProd)(10);

    // shifted Legendre polynomial
    assert(zip((onbs[0] / sqrt(2*0 + 1.0)).coefs, [1]).all!(a => isClose(a[0], a[1]))());
    assert(zip((onbs[1] / sqrt(2*1 + 1.0)).coefs, [-1, 2]).all!(a => isClose(a[0], a[1]))());
    assert(zip((onbs[2] / sqrt(2*2 + 1.0)).coefs, [1, -6, 6]).all!(a => isClose(a[0], a[1]))());
    assert(zip((onbs[3] / sqrt(2*3 + 1.0)).coefs, [-1, 12, -30, 20]).all!(a => isClose(a[0], a[1]))());
}


ElementType!Functions[] gramSchmidt(alias innerProd, Functions)(Functions funcs)
if(isInputRange!Functions && !isInfinite!Functions)
{
    alias F = ElementType!Functions;
    F[] dst;
    if(funcs.empty) return dst;

    foreach(newbasis; funcs) {
        foreach(e; dst) {
            auto c = innerProd(newbasis, e);
            newbasis -= c * e;
        }

        auto norm = sqrt(innerProd(newbasis, newbasis));
        newbasis /= norm;
        dst ~= newbasis;
    }

    return dst;
}


struct NLAnalysisResult(C)
{
    immutable(C)[] coefs;
    typeof(C.init.re) totPower;
    typeof(C.init.re) nlPower;
    typeof(C.init.re) linPower;
    typeof(C.init.re) SDRdB;
}


auto analysisNLCoefs(alias innerProd, C = Complex!double, F, G)(F x, G[] orthonormals)
{
    immutable(C)[] coefs;
    foreach(e; orthonormals) {
        coefs ~= innerProd(x, e);
    }

    auto totpower = innerProd(x, x).re;
    auto linearpower = coefs[0].sqAbs;
    auto nlpower = totpower - linearpower;

    auto SDRdB = 10*std.math.log10(linearpower / nlpower);

    NLAnalysisResult!C dst;
    dst.coefs = coefs;
    dst.SDRdB = SDRdB;
    dst.totPower = totpower;
    dst.linPower = linearpower;
    dst.nlPower = nlpower;

    return dst;
}
