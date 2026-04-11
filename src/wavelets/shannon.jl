struct ShannonWavelet <: Lux.AbstractLuxLayer
    in_dims::Int
    out_dims::Int
end

function Lux.initialparameters(rng::AbstractRNG, l::ShannonWavelet)
    return (weights = Lux.kaiming_uniform(rng, l.in_dims, l.out_dims),)
end

function _reactant_sinc(x)
    mask = (x == zero(x)) * 1.0f0
    safe_x = x + mask
    pix = Float32(pi) * safe_x
    return (one(x) - mask) * sin(pix) / pix + mask
end

function (l::ShannonWavelet)(x, ps, st)
    y = batch_mul(_reactant_sinc.(x .* Float32(2pi)), cos.(x .* Float32(pi / 3))) .* 2.0f0
    return node_mul(y, ps.weights), st
end
