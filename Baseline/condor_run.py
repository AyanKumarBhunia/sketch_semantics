import os
from condor import condor, Job, Configuration
from datetime import datetime
from utils import send_email

conf = Configuration(universe='docker',  # OR 'vanilla'
                     docker_image='registry.eps.surrey.ac.uk/deeprepo:as3',
                     request_CPUs=8,
                     request_memory=1024 * 32,
                     request_GPUs=1,
                     gpu_memory_range=[10000, 24000],
                     cuda_capability=5.5,
                     # following two lists must not overlap
                     # not allowed to run on these
                     restricted_machines=['aisurrey01.surrey.ac.uk'],
                     # allowed_machines=['favmachine.server.com'] # can ONLY run on these machines
                     )
with condor('aisurrey-condor', project_space='sketchCV') as sess:  # aisurrey-condor vs condor

    python_compiler = '/opt/conda/bin/python'    # this is my docker file
    # python_compiler = '/vol/research/sketchCV/anaconda3/envs/sainCVnew/bin/python'
    run_file = f'{os.getcwd()}/main.py'
    exp_name = 'Stroke_correspondence'
    base_folder = '_'.join([exp_name, datetime.now().strftime("%b-%d_%H:%M:%S")])
    message = 'Executor: Aneeshan Sain\nDirectory: Meta_Aux_FGSBIR/SBIR/SBIR_QD\n'
    args = ['try1']

    for i_exp, arg in enumerate(args):

        tag='basic'
        save_dir = f'./condor_output/{base_folder}'
        j = Job(python_compiler, run_file,
                stream_output=True,
                can_checkpoint=True,
                approx_runtime=8,  # in hours
                artifact_dir=save_dir,
                arguments=dict(
                    base_dir=os.getcwd(),
                    saved_models=os.path.join(os.getcwd(), save_dir),
                    disable_tqdm='',
                    max_epoch=200,
                    learning_rate=0.0001
                    )
                , tag=tag+'_')

        sess.submit(j, conf)
        print(f'Job {i_exp + 1}  submitted')

        # Report the job submission
        # message += f'\n\nExperiment {i_exp + 1} :\n' + '\n'.join([f'{item[0]} -- {item[1]}' for item in arg.items()])

    print(message)
    send_email('saneeshan95@gmail.com', message)

print('\nDone')
