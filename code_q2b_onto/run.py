import main


# args=main.parse_args(['--mean_gen','--tbox','--do_train','--do_valid','--do_test','--data_path','lubm_train_cert_w_gen_test_cert_hard','-n',' 128','-b', '512', '-d', '400', '-g', '24', '-lr', '0.0001',
#                            '--max_steps', '2','--cpu_num', '0', '--geo', 'box', '--valid_steps', '15000', '-boxm', '(none,0.02)','--tasks', '1p.2i','--print_on_screen'])
#
#args=main.parse_args(['--mean_gen_new','--tbox','--do_train','--do_valid','--do_test','--data_path','lubm_train_cert_w_gen_test_cert_hard','-n',' 128','-b', '512', '-d', '400', '-g', '24', '-lr', '0.0001',
#                           '--max_steps', '2','--cpu_num', '0', '--geo', 'box', '--valid_steps', '15000', '-boxm', '(none,0.02)','--tasks', '1p.2i','--print_on_screen'])

args=main.parse_args(['--do_test','--data_path', 'data/nell/test_cert_hard','-n',' 128','-b', '512', '-d', '400', '-g', '24', '-lr', '0.0001', '--test_batch_size', '1',
                         '--max_steps', '10','--cpu_num', '0', '--geo', 'box', '--valid_steps', '15000', '-boxm', '(none,0.02)','--tasks', '1p.2p.3p.2i.3i','--print_on_screen',
                      '--checkpoint_path','trained_models/q2b/nell_plain_tr', '--cuda'])


main.main(args)