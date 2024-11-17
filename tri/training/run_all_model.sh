#!/bin/bash
GPU=0
TRAIN_NUM=5
EPOCH=3000
RESULT_DIR=test_result
PATIENCE_LIST=10
ROTATE_LIST=(0.0)
TRANSLATION_LIST=(0.0)
SEED_LIST=(1 42 60)
MODEL_LIST=(prof_input_1_feedback_self_a prof_input_1_feedback_cross_a prof_input_1_feedback model_custom small_unet_cross small_unet_self small_unet model_base )
AUGMODE_LIST=(none)
VERSION=1


for SEED in "${SEED_LIST[@]}"
    do
    for PATIENCE in "${PATIENCE_LIST[@]}"
        do
        for TRANSLATION in "${TRANSLATION_LIST[@]}"
        do
            for  ROTATE in "${ROTATE_LIST[@]}"
                do
                for  AUGMODE in "${AUGMODE_LIST[@]}"
                do
                    if [[(( $AUGMODE == none  &&  $TRANSLATION != 0.0 ) || ( $AUGMODE == none  &&  $ROTATE != 0.0 )) || (( $AUGMODE != none  &&  $TRANSLATION == 0.0 ) && ( $AUGMODE != none  &&  $ROTATE == 0.0 ))]]; then
                        continue
                    fi
                    FILENAME=M${AUGMODE}_T${TRANSLATION}_R${ROTATE}_S${SEED}_P${PATIENCE}_E${EPOCH}_V${VERSION}
                    echo "========================================================"
                    echo "$FILENAME"
                    echo "========================================================"
                    for i in $(seq 1 1 ${TRAIN_NUM})
                    do

                        for MODEL in "${MODEL_LIST[@]}"
                        do
                            python main_tri_repo_prof.py --model ${MODEL}  --trial "$i" --GPU ${GPU}  --result_dir ${RESULT_DIR} --augmode ${AUGMODE} --patience ${PATIENCE} --augrotate ${ROTATE} --augtranslation ${TRANSLATION} --seed ${SEED} --epoch ${EPOCH} --version ${VERSION}
                            
                        done
                    done

                    python get_record.py
                    python get_avg.py
                    python get_rank.py
                    python get_table.py
                    mv csv_record.csv tri_${FILENAME}_record.csv
                    mv avg_record.csv tri_${FILENAME}_avg.csv
                    mv rank_record.csv tri_${FILENAME}_rank.csv
                    mv table_record.csv tri_${FILENAME}_table.csv
                    mv tri_${FILENAME}_record.csv ${RESULT_DIR}
                    mv tri_${FILENAME}_avg.csv ${RESULT_DIR}
                    mv tri_${FILENAME}_rank.csv ${RESULT_DIR}
                    mv tri_${FILENAME}_table.csv ${RESULT_DIR}
                    mv ${RESULT_DIR} ${FILENAME}  
                    cp main_tri_repo_prof.py ${FILENAME}/main.py
                    cp model_custom.py ${FILENAME}/custom.py
                    cp run_all_model.sh ${FILENAME}/run_all_model.sh
                    cp prof_input_1_feedback_self_a.py ${FILENAME}/feedback_self.py
                    cp prof_input_1_feedback_cross_a.py ${FILENAME}/feedback_cross.py
                    cp prof_input_1_feedback.py ${FILENAME}/feedback.py
                    mv ${FILENAME} result

                done
            done
        done
    done
done