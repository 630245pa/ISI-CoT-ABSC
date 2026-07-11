import os
from dotenv import load_dotenv
load_dotenv(dotenv_path="/Users/pranathi/Library/CloudStorage/OneDrive-ErasmusUniversityRotterdam/Masters/Thesis/code_draft/Thesis_draft/.env")  # must be first
print("Key loaded:", os.environ.get("GROQ_API_KEY") is not None)

from data_loader import DataLoader
from sentiment_predictor import SentimentPredictor
from eval import Evaluator, Runner


def main():
    loader = DataLoader(file_path="/Users/pranathi/Downloads/Restaurants_Test_Gold_Implicit_Labeled.xml")
    #loader = DataLoader(file_path="/Users/pranathi/Downloads/Laptops_Test_Gold_Implicit_Labeled.xml")
    data   = loader.load_data()
    loader.print_stats()

    predictor = SentimentPredictor(model="llama3.2") ##change 1
    #predictor = SentimentPredictor(model="llama-4-scout-17b-16e-instruct")
    #predictor = SentimentPredictor(model="llama-3.1-8b-instant")
    #predictor = SentimentPredictor(model="mistral")
    #runner    = Runner(predictor, output_dir="/Users/pranathi/Library/CloudStorage/OneDrive-ErasmusUniversityRotterdam/Masters/Thesis/code_draft/Thesis_draft/results_big_llama_rest/meta-llama")
    #runner    = Runner(predictor, output_dir="/Users/pranathi/Library/CloudStorage/OneDrive-ErasmusUniversityRotterdam/Masters/Thesis/code_draft/Thesis_draft/results_mistral_rest")
    #runner    = Runner(predictor, output_dir="/Users/pranathi/Library/CloudStorage/OneDrive-ErasmusUniversityRotterdam/Masters/Thesis/code_draft/Thesis_draft/results_mistral_laptops")
    runner    = Runner(predictor, output_dir="/Users/pranathi/Library/CloudStorage/OneDrive-ErasmusUniversityRotterdam/Masters/Thesis/code_draft/Thesis_draft/results_lamma_rest")
    evaluator = Evaluator()

    methods = ["zs_scot_si"]
    #methods     = ["vanilla","vanilla_si","zs_cot","zs_cot_si","zs_scot","zs_hcot","zs_scot_si"]  # add more when ready
    #["vanilla", "zs_scot", "zs_hcot", "zs_cot_si", fs_cot]
    # all_results = runner.run_all(data, methods=methods)


    all_results = runner.run_all_parallel(data, methods=methods, max_workers=1)
    evaluator.refinement_stats(all_results["zs_scot_si"], label="zs_scot_si")

 
    # all_results = runner.load_all_results(methods)
    print("\n")
    evaluator.print_compare(all_results)

    for method, results in all_results.items():
        print()
        evaluator.print_by_split(results, label=method, splits=True)
        evaluator.iteration_stats(results, label=method)
        print()
        print(evaluator.report(results))
    
    predictor.token_summary()


if __name__ == "__main__":
    main()



# from sentiment_predictor import SentimentPredictor

# predictor = SentimentPredictor(model="llama3.2")

# text = "The restaurant was super clean."
# print(predictor._get_conllu(text))





    
