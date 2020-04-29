
python test_tsne_raw.py --train_set raw_all \
					--test_set raw_all \
       				--num_out_classes 5 \
					--batch_size 30 \
					--loss_type centerloss \
          --doTSNE \
					--feature_root_folder /home/frank/center/features/centerloss_track_test \
					--info_folder /home/frank/center/test_info/center_raw_track_test \
					--dataset_folder /home/frank/Face_all/test_raw/ | tee -a sreenlog_centerraw_withTrack_test.0