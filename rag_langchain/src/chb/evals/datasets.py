
from mlflow.genai.datasets import create_dataset

def get_dataset():
    

    # Create dataset with manual test cases
    dataset = create_dataset(
        name="eval_rag_dataset2",
        # experiment_id=["0", "1"],
        tags={"dataset_name": "0"},
    )

    # see docs for expected stracture:
    #  https://mlflow.org/docs/latest/genai/concepts/evaluation-datasets/#record-structure
    samples = [
        {
            "inputs": {
                "question": "What were the primary drivers for the decrease in Net revenues for the six months ended June 30, 2025 compared to the same period in 2024?"
            },
            "expectations": {
                "answer": "Net revenues decreased by 12.7% primarily due to lower shipment volumes in North America and Enlarged Europe, unfavorable foreign exchange effects, and mix impacts. [cite_start]Specific operational drivers included a €7.457 billion negative impact from Volume & Mix, a €1.912 billion negative impact from Vehicle Net Price, and a €1.387 billion negative impact from FX and Other."
            },
            "tags": {"test_name": "test1"},
        },
        {
            "inputs": {
                "question": "Why did Stellantis decide to discontinue its hydrogen fuel cell technology development program in 2025?"
            },
            "expectations": {
                "answer": "Management concluded that due to limited availability of hydrogen refueling infrastructure, high capital requirements, and the need for stronger consumer purchasing incentives, the adoption of hydrogen-powered light commercial vehicles would not happen before the end of the decade. [cite_start]Consequently, they discontinued the program, resulting in significant impairments and charges totaling over €733 million, including the impairment of the Symbio joint venture."
            },
            "tags": {"test_name": "test2"},
        },
        {
            "inputs": {
                "question": "What were the Net Revenues for Stellantis in H1 2025 compared to H1 2024?"
            },
            "expectations": {
                "answer": "- H1 2025 Net revenues were €74,261 million\n- H1 2024 Net revenues were €85,017 million\n- This represents a decrease of 12.7%"
            },
            "tags": {"test_name": "test3"},
        },
        {
            "inputs": {
                "question": "Why did Stellantis discontinue its hydrogen fuel cell technology program in 2025?"
            },
            "expectations": {
                "answer": "- Limited availability of hydrogen refueling infrastructure\n- High capital requirements\n- The need for stronger consumer purchasing incentives"
            },
            "tags": {"test_name": "test4"},
        },
        {
            "inputs": {
                "question": "What factors caused the decrease in North America's Adjusted Operating Income in H1 2025?"
            },
            "expectations": {
                "answer": "- Significant unfavorable impacts from volume and mix\n- Increased sales incentives\n- Unfavorable variable cost absorption and warranty costs"
            },
            "tags": {"test_name": "test5"},
        },
        {
            "inputs": {
                "question": "How did the 'One Big Beautiful Bill Act' (OBBB) impact Stellantis' CAFE penalty provisions?"
            },
            "expectations": {
                "answer": "- The act eliminated CAFE fines/penalties (revised rate to $0.00)\n- Resulted in a net expense of €269 million\n- Comprised of: impairment of regulatory credit assets (€609m), onerous contracts (€504m), offset by elimination of CAFE provision (€844m) [cite: 1338, 1339]"
            },
            "tags": {"test_name": "test6"},
        },
        {
            "inputs": {
                "question": "What was the Industrial Free Cash Flow for the first half of 2025?"
            },
            "expectations": {
                "answer": "- Net cash absorption (negative) of €3,005 million\n- This is a decrease of €2,613 million compared to H1 2024"
            },
            "tags": {"test_name": "test7"},
        },
    ]

    samples = [{
            "inputs": {
                "question": "What were the primary drivers for the decrease in Net revenues for the six months ended June 30, 2025 compared to the same period in 2024?"
            },
            "expectations": {
                "answer": "Net revenues decreased by 12.7% primarily due to lower shipment volumes in North America and Enlarged Europe, unfavorable foreign exchange effects, and mix impacts. [cite_start]Specific operational drivers included a €7.457 billion negative impact from Volume & Mix, a €1.912 billion negative impact from Vehicle Net Price, and a €1.387 billion negative impact from FX and Other."
            },
            "tags": {"test_name": "test1"},
        }]

    dataset.merge_records(samples)