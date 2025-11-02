import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, silhouette_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------------------
# Streamlit Page Setup
# ------------------------------
st.set_page_config(page_title="ML Model Explorer", page_icon="🤖", layout="wide")
st.title("🤖 Machine Learning Model Explorer")
st.markdown("### Upload, Train, and Visualize Machine Learning Models Instantly")

# Sidebar
st.sidebar.header("⚙️ Controls")
learning_type = st.sidebar.radio("Select Learning Type", ("Supervised", "Unsupervised", "Auto EDA"))
uploaded_file = st.sidebar.file_uploader("📂 Upload CSV File", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # Handle categorical columns
    label_encoders = {}
    for col in df.select_dtypes(include=["object"]).columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    # Tabs for clean UI
    tab1, tab2, tab3 = st.tabs(["Supervised Learning", "Unsupervised Learning", "Auto EDA"])

    # ------------------------------
    # SUPERVISED LEARNING TAB
    # ------------------------------
    with tab1:
        st.header("🎯 Supervised Learning")
        st.dataframe(df.head())
        target_col = st.selectbox("Select Target Column", df.columns)

        if target_col:
            X = df.drop(columns=[target_col])
            y = df[target_col]

            if len(y.unique()) < 2:
                st.warning("⚠️ Target column must have at least 2 classes.")
            else:
                scaler = StandardScaler()
                X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

                X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

                model_choice = st.selectbox("Choose Model", ["Decision Tree", "Random Forest", "SVM"])

                if model_choice == "Decision Tree":
                    max_depth = st.slider("Max Depth", 1, 20, 5)
                    model = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
                elif model_choice == "Random Forest":
                    n_estimators = st.slider("Number of Trees", 10, 200, 100)
                    model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
                elif model_choice == "SVM":
                    c_val = st.slider("C (Regularization)", 0.01, 10.0, 1.0)
                    kernel = st.selectbox("Kernel", ["linear", "rbf", "poly"])
                    model = SVC(C=c_val, kernel=kernel)

                if st.button("🚀 Train Model"):
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)

                    acc = accuracy_score(y_test, y_pred)
                    st.success(f"✅ Model Trained! Accuracy: **{acc:.2f}**")

                    st.subheader("📋 Classification Report")
                    report_df = pd.DataFrame(classification_report(y_test, y_pred, output_dict=True)).transpose()
                    st.dataframe(report_df)

                    st.subheader("📉 Confusion Matrix")
                    fig, ax = plt.subplots(figsize=(5,4))
                    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, cmap="Blues", fmt="d", ax=ax)
                    st.pyplot(fig)
                    plt.close(fig)

                    if hasattr(model, "feature_importances_"):
                        st.subheader("📊 Feature Importance")
                        importance = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
                        st.bar_chart(importance.head(10))

    # ------------------------------
    # UNSUPERVISED LEARNING TAB
    # ------------------------------
    with tab2:
        st.header("🧠 Unsupervised Learning")
        scaler = StandardScaler()
        df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

        model_choice = st.selectbox("Choose Model", ["KMeans", "Agglomerative", "DBSCAN"], key="unsup")

        if model_choice == "KMeans":
            k = st.slider("Number of Clusters (k)", 2, 10, 3)
            model = KMeans(n_clusters=k, random_state=42)
        elif model_choice == "Agglomerative":
            n_clusters = st.slider("Number of Clusters", 2, 10, 3)
            model = AgglomerativeClustering(n_clusters=n_clusters)
        elif model_choice == "DBSCAN":
            eps = st.slider("Epsilon (eps)", 0.1, 5.0, 0.5)
            min_samples = st.slider("Min Samples", 2, 20, 5)
            model = DBSCAN(eps=eps, min_samples=min_samples)

        if st.button("🌀 Run Clustering"):
            clusters = model.fit_predict(df_scaled)
            df_scaled["Cluster"] = clusters
            st.success("✅ Clustering Completed!")

            st.subheader("🎨 Cluster Visualization")
            fig, ax = plt.subplots()
            sns.scatterplot(x=df_scaled.iloc[:, 0], y=df_scaled.iloc[:, 1], hue="Cluster", palette="tab10", data=df_scaled, ax=ax)
            st.pyplot(fig)
            plt.close(fig)

            if len(set(clusters)) > 1 and -1 not in clusters:
                score = silhouette_score(df_scaled.drop(columns=["Cluster"]), clusters)
                st.info(f"Silhouette Score: **{score:.2f}**")
            else:
                st.warning("⚠️ Silhouette score not available (single cluster or DBSCAN outliers).")

            st.subheader("📊 Clustered Data Preview")
            st.dataframe(df_scaled.head())

            csv = df_scaled.to_csv(index=False)
            st.download_button("💾 Download Clustered Data", data=csv, file_name="clustered_data.csv", mime="text/csv")

    # ------------------------------
    # AUTO EDA TAB
    # ------------------------------
    with tab3:
        st.header("📊 Automatic EDA Dashboard")
        st.write("Dataset Shape:", df.shape)
        st.write("Data Types:", df.dtypes)
        st.write("### Statistical Summary")
        st.dataframe(df.describe())

        st.write("### Correlation Heatmap")
        fig, ax = plt.subplots(figsize=(8,6))
        sns.heatmap(df.corr(), annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)
        plt.close(fig)

else:
    st.info("👈 Upload your CSV file from the sidebar to start exploring!")

st.markdown("<hr><center>Built by Saqib Ahmed | Hackathon Edition</center>", unsafe_allow_html=True)
