FILENAME=ga_610
#FILENAME_610=${FILENAME}_610

python optimize_610_ga.py
python get_record_ga.py
python get_new_custom_ga.py

mv test_result ${FILENAME}
mv csv_record.csv ${FILENAME}.csv 
mv ${FILENAME}.csv ${FILENAME} 
cp main_610_repo_prof_ga.py ${FILENAME}/main.py
cp optimize_610_ga.py ${FILENAME}/optimize.py
cp run_ga.sh ${FILENAME}/run_ga.sh
mv ga_result.txt ${FILENAME}
mv generation_scores.png ${FILENAME}
cp model_custom.py ${FILENAME}/best_custom.py