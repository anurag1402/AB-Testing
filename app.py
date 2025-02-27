import streamlit as st
import pandas as pd
import numpy as np
import scipy.stats
from scipy.stats import norm
import altair as alt
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
import base64


st.set_page_config(
    page_title="A/B Testing Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)


def local_css():
    st.markdown("""
    <style>
        .main-header {
            font-size: 2.5rem;
            color: #1E88E5;
            margin-bottom: 1rem;
        }
        .sub-header {
            font-size: 1.5rem;
            color: #424242;
            margin-bottom: 1rem;
        }
        .card {
            padding: 1.5rem;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            background-color: #ffffff;
            margin-bottom: 1rem;
        }
        .metric-label {
            font-size: 1.2rem;
            font-weight: bold;
            color: #616161;
        }
        .metric-value {
            font-size: 2rem;
            font-weight: bold;
            color: #1E88E5;
        }
        .stButton>button {
            background-color: #1E88E5;
            color: white;
            border-radius: 5px;
            padding: 0.5rem 1rem;
            border: none;
        }
        .stButton>button:hover {
            background-color: #1565C0;
        }
        .annotation {
            font-size: 0.9rem;
            color: #757575;
            font-style: italic;
        }
        .highlight {
            background-color: #E3F2FD;
            padding: 1rem;
            border-radius: 5px;
            border-left: 5px solid #1E88E5;
        }
    </style>
    """, unsafe_allow_html=True)

local_css()

# Helper Functions
def conversion_rate(conversions, visitors):
    """Returns the conversion rate for a given number of conversions and number of visitors."""
    return (conversions / visitors) * 100

def lift(cra, crb):
    """Returns the relative uplift in conversion rate."""
    return ((crb - cra) / cra) * 100

def std_err(cr, visitors):
    """Returns the standard error of the conversion rate."""
    return np.sqrt((cr / 100 * (1 - cr / 100)) / visitors)

def std_err_diff(sea, seb):
    """Returns the standard error of the sampling distribution difference."""
    return np.sqrt(sea ** 2 + seb ** 2)

def z_score(cra, crb, error):
    """Returns the z-score test statistic."""
    return ((crb - cra) / error) / 100

def p_value(z, hypothesis):
    """Returns the p-value based on the z-score and hypothesis type."""
    if hypothesis == "One-sided" and z < 0:
        return 1 - norm().sf(z)
    elif hypothesis == "One-sided" and z >= 0:
        return norm().sf(z) / 2
    else:
        return norm().sf(z)

def significance(alpha, p):
    """Returns whether the p-value is statistically significant."""
    return "YES" if p < alpha else "NO"

def plot_chart_altair(df):
    """Creates a bar chart using Altair."""
    chart = (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=alt.X("Group:O", axis=alt.Axis(labelAngle=0)),
            y=alt.Y("Conversion:Q", title="Conversion rate (%)"),
            color=alt.Color("Group:N", scale=alt.Scale(
                domain=["Control", "Treatment"],
                range=["#5C7CFA", "#51CF66"]
            ))
        )
        .properties(width=500, height=400)
    )

    chart_text = chart.mark_text(
        align="center", baseline="middle", dy=-10, color="black", fontSize=14
    ).encode(text=alt.Text("Conversion:Q", format=",.3g"))

    return st.altair_chart((chart + chart_text).interactive(), use_container_width=True)

def plot_chart_plotly(df):
    """Creates a more visually appealing bar chart using Plotly."""
    fig = px.bar(
        df,
        x="Group",
        y="Conversion",
        color="Group",
        color_discrete_map={"Control": "#5C7CFA", "Treatment": "#51CF66"},
        labels={"Conversion": "Conversion Rate (%)"},
        text="Conversion"
    )

    fig.update_traces(
        texttemplate='%{text:.3g}%',
        textposition='outside',
        marker_line_width=0,
        opacity=0.8
    )

    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        width=700,
        height=500,
        title_text="Conversion Rate Comparison",
        title_x=0.5,
        title_font=dict(size=24, color="#424242"),
        xaxis=dict(
            title=None,
            tickfont=dict(size=16)
        ),
        yaxis=dict(
            title="Conversion Rate (%)",
            tickfont=dict(size=14),
        ),
        legend_title_text=None,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )

    return st.plotly_chart(fig, use_container_width=True)


def style_negative(v, props=""):
    """Helper function to color text in a DataFrame if it is negative."""
    return props if v < 0 else None

def style_p_value(v, props=""):
    """Helper function to color p-value in DataFrame based on significance."""
    return np.where(v < st.session_state.alpha, "color:green; font-weight:bold", props)

def calculate_significance(
    conversions_a, conversions_b, visitors_a, visitors_b, hypothesis, alpha
):
    """Calculates all metrics to be displayed and stores them as session state variables."""
    st.session_state.cra = conversion_rate(int(conversions_a), int(visitors_a))
    st.session_state.crb = conversion_rate(int(conversions_b), int(visitors_b))
    st.session_state.uplift = lift(st.session_state.cra, st.session_state.crb)
    st.session_state.sea = std_err(st.session_state.cra, float(visitors_a))
    st.session_state.seb = std_err(st.session_state.crb, float(visitors_b))
    st.session_state.sed = std_err_diff(st.session_state.sea, st.session_state.seb)
    st.session_state.z = z_score(
        st.session_state.cra, st.session_state.crb, st.session_state.sed
    )
    st.session_state.p = p_value(st.session_state.z, st.session_state.hypothesis)
    st.session_state.significant = significance(
        st.session_state.alpha, st.session_state.p
    )
    
    # Calculate confidence interval (95%)
    st.session_state.ci_lower = (st.session_state.crb - st.session_state.cra) - 1.96 * st.session_state.sed * 100
    st.session_state.ci_upper = (st.session_state.crb - st.session_state.cra) + 1.96 * st.session_state.sed * 100

def plot_power_analysis(cra, visitors_a, visitors_b, alpha=0.05, effect_sizes=None):
    """Creates a plot showing the statistical power for different effect sizes."""
    if effect_sizes is None:
        effect_sizes = np.linspace(0.5, 5, 10)
    
    powers = []
    for effect in effect_sizes:
        # Calculate expected conversion rate for B given the effect size
        crb = cra * (1 + effect / 100)
        
        # Calculate standard errors
        sea = std_err(cra, visitors_a)
        seb = std_err(crb, visitors_b)
        sed = std_err_diff(sea, seb)
        
        # Calculate z-critical for the given alpha
        z_crit = scipy.stats.norm.ppf(1 - alpha)
        
        # Calculate the non-centrality parameter
        delta = ((crb - cra) / sed) / 100
        
        # Calculate power
        power = 1 - scipy.stats.norm.cdf(z_crit - delta)
        powers.append(power * 100)
    
    # Create power curve
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=effect_sizes,
        y=powers,
        mode='lines+markers',
        line=dict(color='#5C7CFA', width=3),
        marker=dict(size=8, color='#1E88E5'),
        name='Statistical Power'
    ))
    
    fig.add_shape(
        type="line",
        x0=min(effect_sizes),
        y0=80,
        x1=max(effect_sizes),
        y1=80,
        line=dict(
            color="#FF9800",
            width=2,
            dash="dash",
        ),
        name="80% Power"
    )
    
    fig.update_layout(
        title="Statistical Power Analysis",
        title_x=0.5,
        xaxis_title="Effect Size (%)",
        yaxis_title="Power (%)",
        plot_bgcolor='rgba(0,0,0,0)',
        yaxis=dict(range=[0, 100]),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return st.plotly_chart(fig, use_container_width=True)

def create_sample_size_calculator():
    """Creates a sample size calculator based on expected effect size."""
    st.markdown('<div class="sub-header">Sample Size Calculator</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        baseline_cr = st.number_input("Baseline Conversion Rate (%)", 
                                     min_value=0.1, 
                                     max_value=99.9, 
                                     value=st.session_state.get('cra', 3.0),
                                     step=0.1,
                                     format="%.1f")
    
    with col2:
        mde = st.number_input("Minimum Detectable Effect (%)", 
                             min_value=1.0, 
                             max_value=100.0, 
                             value=10.0,
                             step=0.5,
                             format="%.1f")
    
    with col3:
        power = st.number_input("Statistical Power (%)", 
                               min_value=50.0, 
                               max_value=99.9, 
                               value=80.0,
                               step=5.0,
                               format="%.1f")
    
    significance_level = st.session_state.get('alpha', 0.05)
    
    # Calculate sample size
    if st.button("Calculate Required Sample Size"):
        # Convert percentages to proportions
        p1 = baseline_cr / 100
        p2 = p1 * (1 + mde / 100)
        
        # Calculate pooled proportion
        p_pooled = (p1 + p2) / 2
        
        # Calculate standard deviations
        sd1 = np.sqrt(p1 * (1 - p1))
        sd2 = np.sqrt(p2 * (1 - p2))
        
        # Calculate z-scores
        z_alpha = scipy.stats.norm.ppf(1 - significance_level / 2)
        z_beta = scipy.stats.norm.ppf(power / 100)
        
        # Calculate sample size per group
        n_per_group = (((z_alpha * np.sqrt(2 * p_pooled * (1 - p_pooled))) + 
                        (z_beta * np.sqrt(p1 * (1 - p1) + p2 * (1 - p2)))) / 
                       (p2 - p1)) ** 2
        
        n_per_group = int(np.ceil(n_per_group))
        
        st.markdown(f"""
        <div class="highlight">
            <div class="metric-label">Required Sample Size:</div>
            <div class="metric-value">{n_per_group:,} per group</div>
            <div class="annotation">Total: {n_per_group * 2:,} visitors</div>
        </div>
        """, unsafe_allow_html=True)

def display_interpretation():
    """Displays an interpretation of the A/B test results."""
    st.markdown('<div class="sub-header">Interpretation</div>', unsafe_allow_html=True)
    
    if st.session_state.significant == "YES":
        result_text = f"""
        <div class="card">
        <p><strong>Result:</strong> This A/B test shows a <span style="color:green">statistically significant</span> 
            difference between the control and treatment groups.</p>
            
        <p><strong>Delta:</strong> The treatment group had a conversion rate of 
            {st.session_state.crb:.3g}%, compared to {st.session_state.cra:.3g}% for the control group, 
            resulting in a <strong>{st.session_state.uplift:.2f}%</strong> uplift.</p>
            
        <p><strong>Confidence:</strong> With a p-value of {st.session_state.p:.4g} (below the 
            significance level of {st.session_state.alpha}), we can be confident that this improvement is 
            not due to random chance.</p>
            
        <p><strong>95% Confidence Interval:</strong> The true improvement is likely between 
            <strong>{st.session_state.ci_lower:.3g}%</strong> and <strong>{st.session_state.ci_upper:.3g}%</strong> 
            (absolute percentage points).</p>
            
        <p><strong>Recommendation:</strong> Consider implementing the treatment version as it shows improved performance.</p>
        </div>
        """
    else:
        result_text = f"""
        <div class="card">
            <p><strong>Result:</strong> This A/B test does <span style="color:red">not show</span> a statistically 
            significant difference between the control and treatment groups.</p>
            
            <p><strong>Delta:</strong> The treatment group had a conversion rate of 
            {st.session_state.crb:.3g}%, compared to {st.session_state.cra:.3g}% for the control group, 
            resulting in a <strong>{st.session_state.uplift:.2f}%</strong> change.</p>
            
            <p><strong>Confidence:</strong> With a p-value of {st.session_state.p:.4g} (above the 
            significance level of {st.session_state.alpha}), the observed difference could be due to random chance.</p>
            
            <p><strong>95% Confidence Interval:</strong> The true difference is likely between 
            <strong>{st.session_state.ci_lower:.3g}%</strong> and <strong>{st.session_state.ci_upper:.3g}%</strong> 
            (absolute percentage points).</p>
            
            <p><strong>Recommendation:</strong> Consider either:</p>
            <ul>
                <li>Running the test longer to collect more data</li>
                <li>Making more substantial changes to the treatment version</li>
                <li>Keeping the control version as the default</li>
            </ul>
        </div>
        """
    
    st.markdown(result_text, unsafe_allow_html=True)

# App Header
st.markdown('<div class="main-header">📊 A/B Testing Dashboard</div>', unsafe_allow_html=True)

st.markdown("""
<div class="card">
    <p>Upload your experiment results to analyze the significance of your A/B test. 
    This app calculates key metrics like conversion rates, uplift, statistical significance, 
    and provides visualization and interpretation of your results.</p>
</div>
""", unsafe_allow_html=True)

# Sidebar for data upload and configuration
with st.sidebar:
    st.markdown('<div class="sub-header">Data Input</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload CSV", type=".csv")
    
    use_example_file = st.checkbox(
        "Use example file", True, help="Use in-built example file to demo the app"
    )
    
    st.markdown('<div class="sub-header">About</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="card">
        <p>This A/B testing dashboard helps you analyze experiment results and make data-driven decisions.</p>
        <p>Features:</p>
        <ul>
            <li>Statistical significance testing</li>
            <li>Conversion rate visualization</li>
            <li>Uplift calculation</li>
            <li>Power analysis</li>
            <li>Sample size calculator</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# Default values
ab_default = None
result_default = None

# If checkbox is filled, use values from the example file
if use_example_file:
    uploaded_file = "Website_Results.csv"
    ab_default = ["variant"]
    result_default = ["converted"]

# Main content area with tabs
# Main content area with tabs
if uploaded_file:
    # Create example data if using the example file
    if uploaded_file == "Website_Results.csv":
        df = pd.DataFrame({
            'variant': ['A'] * 5000 + ['B'] * 5000,
            'converted': [1] * 250 + [0] * 4750 + [1] * 300 + [0] * 4700,
            'timestamp': pd.date_range(start='2025-01-01', periods=10000, freq='H')
        })
    else:
        df = pd.read_csv(uploaded_file)

    st.markdown('<div class="sub-header">Data Preview</div>', unsafe_allow_html=True)
    st.dataframe(df.head(), use_container_width=True)

    # Create tabs for different sections
    tab1, tab2, tab3 = st.tabs(["Analysis", "Visualization", "Planning"])

    with tab1:
        st.markdown('<div class="sub-header">Test Configuration</div>', unsafe_allow_html=True)

        with st.form(key="analysis_form"):
            col1, col2 = st.columns(2)

            with col1:
                ab = st.multiselect(
                    "A/B column",
                    options=df.columns,
                    help="Select which column contains your A/B testing variants",
                    default=ab_default,
                )

                result = st.multiselect(
                    "Result column",
                    options=df.columns,
                    help="Select which column shows the conversion result (1/0)",
                    default=result_default,
                )

            with col2:
                if ab:
                    control = df[ab[0]].unique()[0]
                    treatment = df[ab[0]].unique()[1]
                    decide = st.radio(
                        f"Is *{treatment}* the Treatment Group?",
                        options=["Yes", "No"],
                        help="Select 'Yes' if this is the treatment/variant group"
                    )
                    if decide == "No":
                        control, treatment = treatment, control

                st.markdown('<div class="sub-header">Test Parameters</div>', unsafe_allow_html=True)
                hypothesis = st.radio(
                    "Hypothesis type",
                    options=["One-sided", "Two-sided"],
                    index=0,
                    key="hypothesis",
                    help="One-sided tests if treatment is better than control. Two-sided tests if there's any difference."
                )

                alpha = st.slider(
                    "Significance level (α)",
                    min_value=0.01,
                    max_value=0.10,
                    value=0.05,
                    step=0.01,
                    key="alpha",
                    help="The probability of falsely rejecting the null hypothesis (Type I error)"
                )

            submit_button = st.form_submit_button(label="Run Analysis")

        if not ab or not result:
            st.warning("Please select both an **A/B column** and a **Result column**.")
            st.stop()

        if ab and result and submit_button:
            visitors_a = df[ab[0]].value_counts()[control]
            visitors_b = df[ab[0]].value_counts()[treatment]

            conversions_a = (
                df[[ab[0], result[0]]].groupby(ab[0]).agg("sum")[result[0]][control]
            )
            conversions_b = (
                df[[ab[0], result[0]]].groupby(ab[0]).agg("sum")[result[0]][treatment]
            )

            # Calculate significance metrics
            calculate_significance(
                conversions_a,
                conversions_b,
                visitors_a,
                visitors_b,
                st.session_state.hypothesis,
                st.session_state.alpha,
            )

            # Display summary metrics
            st.markdown('<div class="sub-header">Results Summary</div>', unsafe_allow_html=True)

            col1, col2, col3 = st.columns(3)

            with col1:
                delta = st.session_state.crb - st.session_state.cra
                delta_color = "green" if delta > 0 and st.session_state.significant == "YES" else "red"

                st.markdown(f"""
                <div class="card" style="text-align: center;">
                    <div class="metric-label">Conversion Lift</div>
                    <div class="metric-value" style="color: {delta_color};">{st.session_state.uplift:.2f}%</div>
                    <div class="annotation">Relative improvement</div>
                </div>
                """, unsafe_allow_html=True)

            with col2:
                sig_color = "green" if st.session_state.significant == "YES" else "red"
                sig_text = "Significant" if st.session_state.significant == "YES" else "Not Significant"

                st.markdown(f"""
                <div class="card" style="text-align: center;">
                    <div class="metric-label">Statistical Significance</div>
                    <div class="metric-value" style="color: {sig_color};">{sig_text}</div>
                    <div class="annotation">p-value: {st.session_state.p:.4g}</div>
                </div>
                """, unsafe_allow_html=True)

            with col3:
                st.markdown(f"""
                <div class="card" style="text-align: center;">
                    <div class="metric-label">Sample Size</div>
                    <div class="metric-value">{visitors_a + visitors_b:,}</div>
                    <div class="annotation">Total visitors</div>
                </div>
                """, unsafe_allow_html=True)

            # Display detailed tables
            col1, col2 = st.columns([2, 1])

            with col1:
                table = pd.DataFrame(
                    {
                        "Group": ["Control", "Treatment"],
                        "Converted": [conversions_a, conversions_b],
                        "Total Visitors": [visitors_a, visitors_b],
                        "Conversion Rate": [st.session_state.cra, st.session_state.crb],
                    }
                )

                st.dataframe(
                    table.style.format(
                        formatter={
                            "Conversion Rate": "{:.3g}%",
                            "Converted": "{:,}",
                            "Total Visitors": "{:,}"
                        }
                    ),
                    use_container_width=True
                )

            with col2:
                metrics = pd.DataFrame(
                    {
                        "Metric": ["p-value", "z-score", "Uplift"],
                        "Value": [
                            f"{st.session_state.p:.4g}",
                            f"{st.session_state.z:.3g}",
                            f"{st.session_state.uplift:.2f}%"
                        ]
                    }
                )

                st.dataframe(metrics, use_container_width=True)

            # Display interpretation
            display_interpretation()

    with tab2:
        if 'cra' in st.session_state:
            st.markdown('<div class="sub-header">Conversion Rate Comparison</div>', unsafe_allow_html=True)

            # Create results DataFrame for visualization
            results_df = pd.DataFrame(
                {
                    "Group": ["Control", "Treatment"],
                    "Conversion": [st.session_state.cra, st.session_state.crb],
                }
            )

            # Display chart with Plotly for better visualization
            plot_chart_plotly(results_df)

            # Display confidence interval visualization
            st.markdown('<div class="sub-header">Confidence Interval (95%)</div>', unsafe_allow_html=True)

            fig = go.Figure()

            fig.add_trace(go.Scatter(
                x=[0, 0],
                y=[0, 1],
                mode='markers',
                marker=dict(
                    color='rgba(0,0,0,0)',
                    size=0
                ),
                showlegend=False
            ))

            # Add CI area
            fig.add_shape(
                type="rect",
                x0=st.session_state.ci_lower,
                y0=0.4,
                x1=st.session_state.ci_upper,
                y1=0.6,
                fillcolor="rgba(92, 124, 250, 0.3)",
                line=dict(color="rgba(92, 124, 250, 0.8)", width=1),
            )

            # Add center point (observed difference)
            fig.add_trace(go.Scatter(
                x=[st.session_state.crb - st.session_state.cra],
                y=[0.5],
                mode='markers',
                marker=dict(
                    color='#1E88E5',
                    size=12,
                    symbol='diamond'
                ),
                name='Observed Difference'
            ))

            # Add vertical line at 0 (no effect)
            fig.add_shape(
                type="line",
                x0=0, y0=0, x1=0, y1=1,
                line=dict(
                    color="#616161",
                    width=2,
                    dash="dash",
                ),
            )

            fig.update_layout(
                title="95% Confidence Interval for Conversion Rate Difference",
                title_x=0.5,
                xaxis_title="Absolute Difference in Conversion Rate (%)",
                xaxis=dict(
                    range=[
                        min(st.session_state.ci_lower * 1.5, -0.5),
                        max(st.session_state.ci_upper * 1.5, 0.5)
                    ]
                ),
                yaxis=dict(
                    showticklabels=False,
                    showgrid=False,
                    zeroline=False
                ),
                plot_bgcolor='rgba(0,0,0,0)',
                annotations=[
                    dict(
                        x=st.session_state.ci_lower,
                        y=0.3,
                        text=f"{st.session_state.ci_lower:.3g}%",
                        showarrow=False
                    ),
                    dict(
                        x=st.session_state.ci_upper,
                        y=0.3,
                        text=f"{st.session_state.ci_upper:.3g}%",
                        showarrow=False
                    ),
                    dict(
                        x=st.session_state.crb - st.session_state.cra,
                        y=0.7,
                        text=f"{st.session_state.crb - st.session_state.cra:.3g}%",
                        showarrow=False,
                        font=dict(size=14, color="#1E88E5")
                    )
                ]
            )

            st.plotly_chart(fig, use_container_width=True)

            st.markdown("""
            <div class="card">
                <p>
                <strong>How to interpret:</strong> The confidence interval shows the range in which the true difference
                in conversion rate likely falls with 95% confidence. If the interval includes zero (vertical dashed line),
                the result is not statistically significant.
                </p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info("Run the analysis in the 'Analysis' tab first to see visualizations")

    with tab3:
        if 'cra' in st.session_state:
            st.markdown('<div class="sub-header">Power Analysis</div>', unsafe_allow_html=True)

            st.markdown("""
            <div class="card">
                <p>
                Power analysis helps determine if your test has enough sample size to detect meaningful differences.
                The chart below shows the probability of detecting various effect sizes with your current sample size.
                </p>
            </div>
            """, unsafe_allow_html=True)

            # Custom effect sizes centered around the observed effect
            observed_effect = abs(st.session_state.uplift)
            effect_sizes = np.linspace(max(0.5, observed_effect/2), observed_effect*2, 10)

            plot_power_analysis(
                st.session_state.cra,
                visitors_a,
                visitors_b,
                st.session_state.alpha,
                effect_sizes
            )

            # Sample size calculator
            create_sample_size_calculator()
        else:
            st.info("Run the analysis in the 'Analysis' tab first to use the planning tools")
