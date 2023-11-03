# LUBM
CUDA_VISIBLE_DEVICES=0 python main.py --cuda --do_train --do_valid --do_test \
  --data_path data/lubm/train_cert_test_cert_hard -n 128 -b 512 -d 400 -g 24 \
  -lr 0.0001 --max_steps 450001 --cpu_num 1 --geo box --valid_steps 15000 \
  -boxm "(none,0.02)" --tasks "1p.2p.3p.2i.3i.ip.pi.2u.up"


# NELL
CUDA_VISIBLE_DEVICES=0 python main.py --cuda --do_train --do_valid --do_test \
  --data_path data/nell/train_cert_test_cert_hard -n 128 -b 512 -d 400 -g 24 \
  -lr 0.0001 --max_steps 450001 --cpu_num 1 --geo box --valid_steps 15000 \
  -boxm "(none,0.02)" --tasks "1p.2p.3p.2i.3i.ip.pi.2u.up"


## Rewriting using pre-trained model

CUDA_VISIBLE_DEVICES=0 python main.py --cuda --do_test --query_rew \
  --data_path data/lubm/test_cert -n 128 -b 512 -d 400 -g 24 \
  -lr 0.0001 --max_steps 450001 --cpu_num 1 --geo box --valid_steps 15000 \
  -boxm "(none,0.02)" --tasks "1p.2p.3p.2i.3i.ip.pi.2u.up" --checkpoint_path trained_models/lubm_plain_tr
