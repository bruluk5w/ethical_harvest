from bokeh.models import CustomJSTransform

shift_left = CustomJSTransform(v_func="""
    const shifted = new Float64Array(xs.length);
    for (let i = 0; i < xs.length; i++) {
        shifted[i] = shifted[i] - 0.5;
    }
    return shifted;
    """)

shift_right = CustomJSTransform(v_func="""
    const shifted = new Float64Array(xs.length);
    for (let i = 0; i < xs.length; i++) {
        shifted[i] = shifted[i] + 0.5;
    }
    return shifted;
    """)

mean = CustomJSTransform(v_func="""
    if (xs.length > 0) {
        let average = xs.reduce((a, b) => a + b) / xs.length;
        return Array(xs.length).fill(average);
    } else { 
        return xs;
    }
    """)
