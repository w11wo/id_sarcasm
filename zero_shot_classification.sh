python scripts/run_zero_shot_classification.py --base_model bigscience/bloomz-560m --dataset_name w11wo/reddit_indonesia_sarcastic --output_folder results
python scripts/run_zero_shot_classification.py --base_model bigscience/bloomz-1b1 --dataset_name w11wo/reddit_indonesia_sarcastic --output_folder results
python scripts/run_zero_shot_classification.py --base_model bigscience/bloomz-1b7 --dataset_name w11wo/reddit_indonesia_sarcastic --output_folder results
python scripts/run_zero_shot_classification.py --base_model bigscience/bloomz-3b --dataset_name w11wo/reddit_indonesia_sarcastic --output_folder results
python scripts/run_zero_shot_classification.py --base_model bigscience/bloomz-7b1 --dataset_name w11wo/reddit_indonesia_sarcastic --output_folder results

python scripts/run_zero_shot_classification.py --base_model bigscience/mt0-small --dataset_name w11wo/reddit_indonesia_sarcastic --output_folder results
python scripts/run_zero_shot_classification.py --base_model bigscience/mt0-base --dataset_name w11wo/reddit_indonesia_sarcastic --output_folder results
python scripts/run_zero_shot_classification.py --base_model bigscience/mt0-large --dataset_name w11wo/reddit_indonesia_sarcastic --output_folder results
python scripts/run_zero_shot_classification.py --base_model bigscience/mt0-xl --dataset_name w11wo/reddit_indonesia_sarcastic --output_folder results
# python scripts/run_zero_shot_classification.py --base_model bigscience/mt0-xxl --dataset_name w11wo/reddit_indonesia_sarcastic --output_folder results

python scripts/run_zero_shot_classification.py --base_model bigscience/bloomz-560m --dataset_name w11wo/twitter_indonesia_sarcastic --text_column_name tweet --output_folder results
python scripts/run_zero_shot_classification.py --base_model bigscience/bloomz-1b1 --dataset_name w11wo/twitter_indonesia_sarcastic --text_column_name tweet --output_folder results
python scripts/run_zero_shot_classification.py --base_model bigscience/bloomz-1b7 --dataset_name w11wo/twitter_indonesia_sarcastic --text_column_name tweet --output_folder results
python scripts/run_zero_shot_classification.py --base_model bigscience/bloomz-3b --dataset_name w11wo/twitter_indonesia_sarcastic --text_column_name tweet --output_folder results
python scripts/run_zero_shot_classification.py --base_model bigscience/bloomz-7b1 --dataset_name w11wo/twitter_indonesia_sarcastic --text_column_name tweet --output_folder results

python scripts/run_zero_shot_classification.py --base_model bigscience/mt0-small --dataset_name w11wo/twitter_indonesia_sarcastic --text_column_name tweet --output_folder results
python scripts/run_zero_shot_classification.py --base_model bigscience/mt0-base --dataset_name w11wo/twitter_indonesia_sarcastic --text_column_name tweet --output_folder results
python scripts/run_zero_shot_classification.py --base_model bigscience/mt0-large --dataset_name w11wo/twitter_indonesia_sarcastic --text_column_name tweet --output_folder results
python scripts/run_zero_shot_classification.py --base_model bigscience/mt0-xl --dataset_name w11wo/twitter_indonesia_sarcastic --text_column_name tweet --output_folder results
# python scripts/run_zero_shot_classification.py --base_model bigscience/mt0-xxl --dataset_name w11wo/reddit_indonesia_sarcastic --text_column_name tweet --output_folder results