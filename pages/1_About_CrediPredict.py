import streamlit as st

st.set_page_config(
    page_title="About CrediPredict",
    page_icon="📊",
    layout="wide"
)

st.title("About CrediPredict")
st.caption("Credible predictions for market-moving events")

st.markdown(
    """
    ## Project Aim

    **CrediPredict** is a financial forecasting project designed to transform market information into structured prediction signals.

    The long-term idea is to create a platform where users can suggest stock-related events and vote on:

    - how important the event is for stock price or volatility;
    - how likely the event is to happen;
    - how the event may affect the stock price;
    - whether the impact is expected in days, weeks, months, or longer horizons.

    The platform will separate predictions from different user groups, such as free voters, verified voters, and premium voters.  
    This allows investors to compare general crowd opinion with more credible and historically accurate forecasters.
    """
)

st.markdown("---")

st.markdown(
    """
    ## Current Prototype

    The current version of CrediPredict is a simplified academic prototype.

    At this stage, the website does **not** use event-based voting yet.  
    Instead, it uses historical stock market data from Yahoo Finance and applies a simple machine-learning model.

    The current prototype:

    - downloads updated stock data;
    - calculates historical returns and volatility;
    - uses price, return, volume, and market benchmark features;
    - predicts whether a selected stock is more likely to go up or down in the next trading day;
    - estimates a next-day price range;
    - compares model accuracy with a baseline strategy.

    The result shows that historical price data alone has limited predictive power.  
    This supports the future development of CrediPredict, where statistical models will be combined with event-based crowd forecasting and credibility scores.
    """
)

st.markdown("---")

st.markdown(
    """
    ## Future Version

    In the future, CrediPredict can include:

    - event suggestion system;
    - community voting;
    - credibility score for users;
    - verified and premium forecaster groups;
    - profile badges for diplomas, certificates, exams, courses, and workplace recommendations;
    - investor subscription access;
    - event-based market intelligence dashboard.

    The goal is not to provide guaranteed financial advice, but to create a structured decision-support system based on data, events, and credible human forecasting.
    """
)

st.markdown("---")

st.markdown("## Join the Community Voting Channel")

st.write(
    """
    We are building a community where users can discuss and vote on market-moving events.
    The WhatsApp Channel will be used to collect early community predictions and test the future event-voting idea.
    """
)

# Replace this with your real WhatsApp Channel link
whatsapp_channel_link = "PASTE_YOUR_WHATSAPP_CHANNEL_LINK_HERE"

st.link_button("Join CrediPredict WhatsApp Channel", whatsapp_channel_link)

st.warning(
    "Disclaimer: CrediPredict is an academic and experimental prototype. It does not provide financial advice."
)
