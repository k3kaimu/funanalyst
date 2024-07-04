module funanalyst.space;

import std.algorithm;
import std.complex;
import std.format;
import std.math;
import std.meta;
import std.range;
import std.traits;

import mir.ndslice : Slice, slice, sliced, ndassign;


private
template NDArrayType(X, uint n)
{
    static if(n == 0)
        alias NDArrayType = X;
    else
        alias NDArrayType = NDArrayType!(X, n-1)[];
}


private
template DeepUnqual(T)
{
    static if(isArray!T)
        alias DeepUnqual = DeepUnqual!(ElementType!T)[];
    else
        alias DeepUnqual = Unqual!T;
}


private
DeepUnqual!T[] deepdup(T)(in T[] arr)
{
    DeepUnqual!T[] dst = new DeepUnqual!T[](arr.length);
    foreach(i, ref e; dst)
        e = arr[i].deepdup;
    
    return dst;
}


private
Unqual!T deepdup(T)(T v)
if(!isArray!T && is(T : Unqual!T))
{
    return v;
}


unittest
{
    int[][] arr1 = [[1, 2], [3, 4]];
    auto arr2 = arr1.deepdup;

    arr2[0][0] = 10;
    assert(arr1[0][0] == 1);
}


struct LinearCombination(X, alias base, uint Dim_ = 1)
if(Dim_ >= 1)
{
    alias basis = base;
    alias ParamType = X;
    enum size_t Dim = Dim_;


    this(in NDArrayType!(X, Dim_) coefs)
    {
        _coefs = coefs.deepdup;
    }


    this(this)
    {
        _coefs = _coefs.deepdup;
    }


    X opCall(U)(U x)
    {
        X dst = X(0);
        // foreach(i, e; _coefs)
        //     dst += e * base(x, i);
        // mixin(iota(Dim).map!(a => format!`foreach(i%1$s; 0 .. this._coefs.length!%1$s) `(a) ).join
        //     ~ "dst += "
        //     ~ "this._coefs[" ~  iota(Dim).map!(a => format!`i%s, `(a)).join ~ "]"
        //     ~ "* base(x, " ~  iota(Dim).map!(a => format!`i%s, `(a)).join ~ ");"
        // );
        mixin(
            genForeaches()
            ~ " dst += this._coefs" ~ genIndecies(0, Dim) ~ "* base(x, " ~ genIndecies(0, Dim, "i%1$s, ") ~ ");"
        );
        // pragma(msg,  genForeaches()
        //     ~ " dst += this._coefs" ~ genIndecies(0, Dim) ~ "* base(x, " ~ genIndecies(0, Dim, "i%1$s, ") ~ ");");

        return dst;
    }


    typeof(this) opBinary(string op)(typeof(this) rhs)
    if(op == "+" || op == "-")
    {
        auto dst = this;
        dst.opOpAssign!op(rhs);
        return dst;
    }


    typeof(this) opBinary(string op, U)(in U scalar)
    if(op == "*" || op == "/")
    {
        auto dst = this;
        dst.opOpAssign!op(scalar);

        return dst;
    }


    typeof(this) opBinaryRight(string op, U)(in U scalar)
    if(op == "*")
    {
        return this.opBinary!op(scalar);
    }


    void opOpAssign(string op)(typeof(this) rhs)
    if(op == "+" || op == "-")
    {
        size_t[Dim] oldshape = this._shape;
        size_t[Dim] newshape, rhsshape;
        // rhsshape = rhs._shape;
        // newshape[] = oldshape[].zip(rhsshape[]).map!(a => max(a[0], a[1])).array[];
        rhsshape = rhs._shape;
        foreach(i; 0 .. Dim) {
            newshape = max(oldshape[i], rhsshape[i]);
        }

        if(newshape != oldshape) {
            this._reshape(newshape);
        }

        mixin(iota(Dim).map!(a => format(`foreach(i%1$s; 0 .. rhsshape[%1$s] )`, a)).join()
        ~ "this._coefs" ~ genIndecies() ~ op ~ "= " ~ "rhs._coefs" ~ genIndecies() ~ ";");
    }


    void opOpAssign(string op, U)(in U scalar)
    if(op == "*" || op == "/")
    {
        mixin("_coefs[] " ~ op ~ "= X(scalar);");
    }


    auto coefs() inout { return _coefs; }


  private:
    NDArrayType!(X, Dim) _coefs;


    static
    string genForeaches(string fmt = `foreach(i%s; 0 .. _coefs%s.length)`)
    {
        string str;
        foreach(i; 0 .. Dim_)
            str ~= format(fmt, i, genIndecies(0, i));

        return str;
    }


    static
    string genIndecies(size_t a = 0, size_t b = Dim, string fmt = `[i%1$s]`)
    {
        string str;
        foreach(i; a .. b)
            str ~= format(fmt, i);
        
        return str;
    }


    size_t[Dim] _shape()
    {
        typeof(return) dst;

        static foreach(i; 0 .. Dim_)
        {
            dst[i] = mixin(`this._coefs`~ "[0]".repeat(i).join ~ ".length");
        }

        return dst;
    }


    void _reshape(size_t[Dim] newshape)
    {
        size_t[Dim] oldshape = this._shape;
        auto newslice = mixin("new NDArrayType!(X, Dim)(" ~ genIndecies(0, Dim, "newshape[%1$s], ") ~ ")");
        mixin(genForeaches(`foreach(i%s; 0 .. newslice%s.length)`) ~ "newslice" ~ genIndecies() ~ " = X(0);");
        mixin(iota(Dim).map!(a => format(`foreach(i%1$s; 0 .. oldshape[%1$s] )`, a)).join() ~ " newslice" ~ genIndecies() ~ " = this._coefs" ~ genIndecies() ~ ";");
        this._coefs = newslice;
    }
}


auto polynomialBasis(X)(X x, long n)
{
    return x^^n;
}


auto widelyPolynomialBasis(X)(X x, long p, long q)
{
    return x^^p * (x.conj)^^q;
}


auto widelyPolynomialOnlyAM(X)(X x, long p)
{
    return x * x.sqAbs^^p;
}


alias Polynomial(X) = LinearCombination!(X, polynomialBasis);

alias WidelyPolynomial(X) = LinearCombination!(X, widelyPolynomialBasis, 2);
alias WidelyPolynomialOnlyAM(X) = LinearCombination!(X, widelyPolynomialOnlyAM, 1);


unittest
{
    auto f1 = Polynomial!int([1, 2, 3]);
    auto f2 = f1;

    assert(f1.coefs == f2.coefs);
    assert(f1.coefs.ptr != f2.coefs.ptr);

    f2 = typeof(f2).init;
    f2 = f1;
    assert(f1.coefs == f2.coefs);
    assert(f1.coefs.ptr != f2.coefs.ptr);
}

unittest
{
    foreach(X; AliasSeq!(int, Complex!double, double))
    {
        // 1 + 2*x + 3*x*x
        auto f1 = Polynomial!X([X(1), X(2), X(3)]);
        assert(f1(0) == 1);
        assert(f1(1) == 6);
        assert(f1(2) == 17);

        // 0 + 1*x + -1*x*x
        auto f2 = Polynomial!X([X(0), X(1), X(-1)]);
        assert(f2(0) == 0);
        assert(f2(1) == 0);
        assert(f2(2) == -2);

        auto f3 = f1 + f2;
        assert(f3(0) == 1);
        assert(f3(1) == 6);
        assert(f3(2) == 15);

        auto f4 = f1 - f2;
        assert(f4(0) == 1);
        assert(f4(1) == 6);
        assert(f4(2) == 19);

        auto f5 = f1 * 2;
        assert(f5(0) == 2);
        assert(f5(1) == 12);
        assert(f5(2) == 34);

        auto f6 = f1 / -1;
        assert(f6(0) == -1);
        assert(f6(1) == -6);
        assert(f6(2) == -17);

        auto f7 = 2 * f1;
        assert(f7(0) == 2);
        assert(f7(1) == 12);
        assert(f7(2) == 34);
    }
}

unittest
{
    import std.stdio;
    import std.complex;
    alias C = Complex!double;

    bool capproxEqual(C x, C y)
    {
        return isClose(x.re, y.re) && isClose(x.im, y.im);
    }

    auto f1 = WidelyPolynomial!C([[C(1), C(2)], [C(3), C(4)]]);

    // writeln(f1);
    // writeln(f1(C(1, 1)));
    // writeln(1 + C(1,1)*2 + C(1,-1)*3 + C(1,1)*C(1,-1)*4);

    assert(f1(C(1, 1)) == 1 + C(1,-1)*2 + C(1,1)*3 + C(1,1)*C(1,-1)*4);
}

unittest
{
    auto f1 = Polynomial!double();
    assert(f1(2) == 0);

    auto f2 = Polynomial!double([1]);
    f1 += f2;
    assert(f1(2) == 1);

    auto f3 = Polynomial!double([0, 1]);
    f1 += f3;
    assert(f1(2) == 3);
}
