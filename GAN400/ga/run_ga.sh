FILENAME=ga_GAN400
#FILENAME_GAN400=${FILENAME}_GAN400

python optimize_GAN400_ga.py
python get_record_ga.py
python get_new_custom_ga.py

mv test_result ${FILENAME}
mv csv_record.csv ${FILENAME}.csv 
mv ${FILENAME}.csv ${FILENAME} 
cp main_GAN400_repo_prof_ga.py ${FILENAME}/main.py
cp optimize_GAN400_ga.py ${FILENAME}/optimize.py
cp run_ga.sh ${FILENAME}/run_ga.sh
mv ga_result.txt ${FILENAME}
mv generation_scores.png ${FILENAME}
cp model_custom.py ${FILENAME}/best_custom.py