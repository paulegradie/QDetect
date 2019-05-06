from argparse import ArgumentParser
import os
import signal


def read_prev_pids(id_file='temp_pid'):
    try:
        with open(id_file, 'r') as pin:
            pid_list = [int(x.strip()) for x in pin.readlines()]
    except IOError:
        pid_list = None
    return pid_list

def kill_prev_pids(pid_list):
    if pid_list:
        try:
            for pid in pid_list:
                os.kill(pid, signal.SIGTERM)
        except Exception as e:
            pass
        return True
    else:
        print('no pids found')
        return False


if __name__ == "__main__":

    parser = ArgumentParser()

    parser.add_argument('-f', '--file', default='temp_pid', help='path to pid file')
    args = parser.parse_args()

    file_path = args.file
    print('Pid file path: {}'.format(file_path))

    pid_list = read_prev_pids(file_path)
    print('pids: {}'.format(pid_list))

    res = kill_prev_pids(pid_list)

    if res:
        os.remove('temp_pid')


    print("Done")
    print("proccesses killed: {}".format(pid_list))