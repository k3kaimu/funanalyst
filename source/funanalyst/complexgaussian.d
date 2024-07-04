module funanalyst.complexgaussian;

import std.math;
import dffdd.math;

import funanalyst.space;



WidelyPolynomialOnlyAM!C[] complexGaussianONBasisFunctionsOnlyAM(C)(size_t maxOrder)
{
    WidelyPolynomialOnlyAM!C[] dst;
    foreach(p; 0 .. (maxOrder+1)/2) {

        C[] coefs;
        foreach(q; 0 .. p+1)
            coefs ~= C(binom!real(p + 1, q + 1) / sqrt(p+1.0) / fact!real(cast(uint)q) * (-1.0L)^^(p + q));
        
        dst ~= WidelyPolynomialOnlyAM!C(coefs);
    }

    return dst;
}


unittest
{
    import std.stdio;

    import std.algorithm;
    import std.range;

    import std.complex : Complex;
    alias C = Complex!double;

    auto funcs = complexGaussianONBasisFunctionsOnlyAM!C(5);
    assert(zip((funcs[0] * sqrt(1.0)).coefs.map!"a.re", [1]).all!(a => isClose(a[0], a[1]))());
    assert(zip((funcs[1] * sqrt(2.0)).coefs.map!"a.re", [-2, 1]).all!(a => isClose(a[0], a[1]))());
    assert(zip((funcs[2] * sqrt(3.0)).coefs.map!"a.re", [3, -3, 0.5]).all!(a => isClose(a[0], a[1]))());
}
