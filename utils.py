#!/usr/bin/env python
# -*- coding: utf-8 -*-

import importlib
import os
import pwd
import socket
import sys
import urllib
import zipfile
from random import randint
from time import gmtime, strftime

import requests
from PIL import Image
from six.moves import urllib as smurllib

import FLAGS
import PARAMS



def check_dependencies_installed():
    """
    Checks whether the needed dependencies are installed.

    :return: a list of missing dependencies
    """
    missing_dependencies = []

    try:
        import importlib
    except ImportError:
        missing_dependencies.append("importlib")

    dependencies = ["termcolor",
                    "colorama",
                    "tensorflow",
                    "numpy",
                    "PIL",
                    "six",
                    "tarfile",
                    "zipfile",
                    "requests"]

    for dependency in dependencies:
        if not can_import(dependency):
            missing_dependencies.append(dependency)

    return missing_dependencies



def can_import(some_module):
    """
    Checks whether a module is installed by trying to import it.

    :param some_module: the name of the module to check

    :return: a boolean representing whether the import is successful.
    """

    try:
        importlib.import_module(some_module)
    except ImportError:
        return False

    return True



def maybe_download_and_extract():
    """Downloads and extracts the zip from electronneutrino, if necessary"""
    dest_directory = FLAGS.data_dir
    if not os.path.exists(dest_directory):
        os.makedirs(dest_directory)
    filename = PARAMS.DATA_URL.split('/')[-1]
    filepath = os.path.join(dest_directory, filename)
    if not os.path.exists(filepath):
        print_progress_bar(0, 100,
                           prefix='Downloading ' + filename + ":", suffix='Complete', length=50,
                           fill='█')


        def _progress(count, block_size, total_size):
            print_progress_bar(float(count * block_size) / float(total_size) * 100.0, 100,
                               prefix='Downloading ' + filename + ":", suffix='Complete', length=50,
                               fill='█')


        filepath, _ = smurllib.request.urlretrieve(PARAMS.DATA_URL, filepath, _progress)
        print()
        statinfo = os.stat(filepath)
        print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')

    extracted_dir_path = os.path.join(dest_directory, 'cifar-10-batches-bin')

    if not os.path.exists(extracted_dir_path):
        zip_ref = zipfile.ZipFile(filepath, 'r')
        zip_ref.extractall(dest_directory)
        zip_ref.close()



def verify_dataset():
    """
    Verifies the authenticity of the dataset.

    :raises: Exception if the dataset's images are the wrong size.

    :return: nothing on success
    """
    which = randint(1, 10000)

    where = os.path.join(FLAGS.data_dir, 'images/%d_L.png' % which)

    im = Image.open(where)
    width, height = im.size

    # print("w, h: " + str(width) + ", " + str(height))

    if not (width == PARAMS.IMAGE_SIZE and height == PARAMS.IMAGE_SIZE):
        raise Exception("Dataset appears to have been corrupted. (Check " + where + ")")



def notify(message, subject="Notification", email=FLAGS.NOTIFICATION_EMAIL):
    """
    Send an email with the specified message.

    :param message: the message to be sent
    :param subject: (optional) the subject of the message
    :param email: (optional) the email to send the message to. Defaults to FLAGS.NOTIFICATION_EMAIL

    :return: The response of the server. Should be "Thanks!"
    """
    # params = {'message': "[" + get_username() + "@" + get_hostname() + ", " + get_time_string() + "]: " + message,
    #           'subject': subject, 'email': email}
    # encoded_params = urllib.urlencode(params)

    # response = requests.get('https://electronneutrino.com/affinity/notify/notify.php?' + encoded_params)
    # # print (response.status_code)
    # # print (response.content)
    response = "Thanks!"

    return response



def get_time_string():
    """
    Returns the GMT.

    :return: a formatted string containing the GMT.
    """
    return strftime("%Y-%m-%d %H:%M:%S", gmtime()) + " GMT"



def get_username():
    """
    Gets the username of the current user.

    :return: a string with the username
    """
    return pwd.getpwuid(os.getuid()).pw_name



def get_hostname():
    """
    Returns the hostname of the computer.

    :return: a string containing the hostname
    """
    return socket.gethostname()



def print_progress_bar(iteration, total, prefix='', suffix='', decimals=1, length=100, fill="█"):
    """
    Call in a loop to create terminal progress bar. Based on https://stackoverflow.com/a/34325723
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    sys.stdout.write('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix) + '\r')
    sys.stdout.flush()
    # Print New Line on Complete
    if iteration == total:
        print()
        print()
        sys.stdout.write("")
