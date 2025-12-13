"""
AI Bias Detector - Main Entry Point
Detects political bias in news articles and verifies their authenticity
"""

import sys
from gemini_handler import GeminiHandler
from bias_detector_hf import BiasDetectorHF
import matplotlib.pyplot as plt


def print_separator():
    print("\n" + "=" * 70 + "\n")


def display_pie_chart(left_score, right_score, neutral_score):
    """Display a pie chart of bias distribution"""
    labels = ['Left', 'Right', 'Neutral']
    sizes = [left_score, right_score, neutral_score]
    colors = ['#3498db', '#e74c3c', '#95a5a6']
    explode = (0.1, 0.1, 0.1)

    plt.figure(figsize=(8, 6))
    plt.pie(
        sizes,
        explode=explode,
        labels=labels,
        colors=colors,
        autopct='%1.1f%%',
        shadow=True,
        startangle=90
    )
    plt.axis('equal')
    plt.title('Political Bias Distribution', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()


def main():
    print("=" * 70)
    print(" " * 20 + "AI BIAS DETECTOR")
    print("=" * 70)

    # Initialize handlers
    try:
        print("\n🔧 Initializing Gemini API...")
        gemini = GeminiHandler()

        print("\n🤖 Loading Hugging Face bias detection model...")
        print("   (This may take a moment on first run - model will be cached)")
        bias_detector = BiasDetectorHF()

        print("\n✅ System initialized successfully!\n")
    except Exception as e:
        print(f"❌ Error initializing system: {str(e)}")
        sys.exit(1)

    while True:
        print_separator()
        print("Enter a news article headline or full article (or 'quit' to exit):")
        print_separator()

        user_input = input("📰 News: ").strip()

        if user_input.lower() in ['quit', 'exit', 'q']:
            print("\n👋 Thank you for using AI Bias Detector!")
            break

        if not user_input:
            print("⚠️  Please enter some text!")
            continue

        print("\n🔍 Processing your input...\n")

        # STEP 1: Verify news
        print("STEP 1: Verifying news authenticity...")
        print("-" * 70)

        verification_result = gemini.verify_news(user_input)

        if verification_result['error']:
            print(f"❌ Error: {verification_result['error']}")
            continue

        print(f"✅ Verification Status: {verification_result['is_true']}")
        print(f"📝 Analysis: {verification_result['analysis']}")

        # STEP 2: Bias detection
        print_separator()
        print("STEP 2: Detecting political bias with ML model...")
        print("-" * 70)

        bias_result = bias_detector.detect_bias(user_input)

        print("\n📊 Bias Detection Results:")
        print(f"   Left Bias:    {bias_result['left']:.2f}%")
        print(f"   Right Bias:   {bias_result['right']:.2f}%")
        print(f"   Neutral:      {bias_result['neutral']:.2f}%")
        print(f"\n🎯 Overall Lean: {bias_result['overall_bias']}")
        print(f"🔬 Detection Method: {bias_result['method']}")

        # STEP 3: Summary
        print_separator()
        print("STEP 3: Generating news summary...")
        print("-" * 70)

        summary = gemini.summarize_news(user_input)
        print(f"\n📄 Summary:\n{summary}\n")

        # STEP 4: Visualization
        print_separator()
        show_chart = input("📊 Would you like to see a pie chart? (y/n): ").strip().lower()

        if show_chart == 'y':
            display_pie_chart(
                bias_result['left'],
                bias_result['right'],
                bias_result['neutral']
            )

        print_separator()
        continue_prompt = input("🔄 Analyze another article? (y/n): ").strip().lower()
        if continue_prompt != 'y':
            print("\n👋 Thank you for using AI Bias Detector!")
            break


if __name__ == "__main__":
    main()
