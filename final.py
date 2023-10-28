import streamlit as st

# Set the background color and text color
st.markdown(
    """
    <style>
    body {
        background-color: #000;
        color: #FFF;
    }
    .sidebar {
        background-color: #000;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Create a left sidebar
st.sidebar.title("Options")

# Add options for different features
selected_option = st.sidebar.radio(
    "Select a Feature", ["Translation", "Querying", "Citation", "Summary"]
)

# Display content based on the selected feature
if selected_option == "Translation":
    st.title("Translation Feature")
    # Add code for the translation feature here
elif selected_option == "Querying":
    st.title("Querying Feature")
    # Add code for the querying feature here
elif selected_option == "Citation":
    st.title("Citation Feature")
    # Add code for the citation feature here
elif selected_option == "Summary":
    st.title("Summary Feature")
    # Add code for the summary feature here
