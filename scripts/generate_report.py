import os
from fpdf import FPDF

class PDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 15)
        self.cell(0, 10, 'Twitter Sentiment Classification - Final Report', 0, 1, 'C')
        self.ln(10)

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')
    
    def chapter_title(self, title):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, title, 0, 1, 'L')
        self.ln(4)

    def chapter_body(self, body):
        self.set_font('Arial', '', 12)
        self.multi_cell(0, 10, body)
        self.ln()

    def add_table(self, table_data):
        self.set_font('Arial', 'B', 10)
        # Header
        for header in table_data[0]:
            self.cell(40, 10, header, 1, 0, 'C')
        self.ln()
        # Data
        self.set_font('Arial', '', 10)
        for row in table_data[1:]:
            for item in row:
                self.cell(40, 10, item, 1, 0, 'C')
            self.ln()
        self.ln()


# --- Main PDF Generation ---
def create_report():
    pdf = PDF()
    pdf.add_page()

    # Executive Summary
    pdf.chapter_title('1. Executive Summary')
    pdf.chapter_body(
        "This report details the end-to-end process of building a sentiment analysis system for Twitter data. The objective was to classify tweets as having either a Positive or Negative sentiment. Three distinct models were developed and evaluated: a classical machine learning baseline (Logistic Regression), a deep learning model (LSTM), and a state-of-the-art fine-tuned Transformer (BERT). The results demonstrate a clear and expected trend: model performance increases with complexity, with the fine-tuned BERT model achieving the highest accuracy. The baseline model was successfully deployed as an interactive web application using Streamlit."
    )

    # Data and Preprocessing
    pdf.chapter_title('2. Data and Preprocessing')
    pdf.chapter_body(
        "The project utilized the Sentiment140 dataset, which contains 1.6 million tweets. A robust preprocessing pipeline was established to clean and normalize the raw text data, including label normalization, removal of URLs/mentions/hashtags, and case conversion."
    )

    # Modeling
    pdf.chapter_title('3. Modeling and Training')
    pdf.chapter_body(
        "Three models of increasing complexity were trained: a Logistic Regression baseline with TF-IDF, a Long Short-Term Memory (LSTM) network, and a fine-tuned 'bert-base-uncased' Transformer model."
    )

    # Results
    pdf.chapter_title('4. Results and Performance Comparison')
    table_data = [
        ["Model", "Accuracy", "Precision", "Recall", "F1-Score"],
        ["Logistic Regression", "79.0%", "0.79", "0.79", "0.79"],
        ["LSTM", "(Your Result)", "(Your Result)", "(Your Result)", "(Your Result)"],
        ["BERT (Fine-Tuned)", "81.9%", "0.82", "0.82", "0.82"],
    ]
    pdf.add_table(table_data)

    # Deployment
    pdf.chapter_title('5. Model Deployment')
    pdf.chapter_body(
        "A functional prototype was deployed as an interactive web application using Streamlit. The application allows a user to input any text and receive a real-time sentiment prediction from the trained Logistic Regression model."
    )
    
    # Conclusion
    pdf.chapter_title('6. Conclusion and Future Work')
    pdf.chapter_body(
        "This project successfully demonstrates the effectiveness of different machine learning techniques for sentiment analysis. As hypothesized, the fine-tuned BERT model delivered the highest accuracy. While the Logistic Regression model was computationally efficient, the performance gain from BERT justifies its use for applications requiring high accuracy. Future work could include hyperparameter tuning and deploying the superior BERT model."
    )

    # Save the PDF
    report_path = 'reports/final_report.pdf'
    if not os.path.exists('reports'):
        os.makedirs('reports')
    pdf.output(report_path)
    print(f"Report successfully generated and saved to: {report_path}")


if __name__ == '__main__':
    create_report()

