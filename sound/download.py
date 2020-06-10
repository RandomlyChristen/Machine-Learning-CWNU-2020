'''
================================================
          DOWNLOAD_AUDIOSET REPOSITORY
================================================

repository name: download_audioset
repository version: 1.0
repository link: https://github.com/jim-schwoebel/download_audioset
author: Jim Schwoebel
author contact: js@neurolex.co
description: downloads the raw audio files from AudioSet (released by Google).
license category: opensource
license: Apache 2.0 license
organization name: NeuroLex Laboratories, Inc.
location: Seattle, WA
website: https://neurolex.ai
release date: 2018-11-08

This code (download_audioset) is hereby released under a Apache 2.0 license license.

For more information, check out the license terms below.

================================================
                SPECIAL NOTES
================================================

This script parses through the entire balanced audioset dataset and downloads
all the raw audio files. The files are arranged in folders according to their
representative classes.

Please ensure that you have roughly 35GB of free space on your computer before
downloading the files. Note that it may take up to 2 days to fully download
all the files.

Enjoy! - :)

-Jim

================================================
                LICENSE TERMS
================================================

Copyright 2018 NeuroLex Laboratories, Inc.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

================================================
                SERVICE STATEMENT
================================================

If you are using the code written for a larger project, we are
happy to consult with you and help you with deployment. Our team
has >10 world experts in Kafka distributed architectures, microservices
built on top of Node.js / Python / Docker, and applying machine learning to
model speech and text data.

We have helped a wide variety of enterprises - small businesses,
researchers, enterprises, and/or independent developers.

If you would like to work with us let us know @ develop@neurolex.co.
'''

################################################################################
##                            IMPORT STATEMENTS                               ##
################################################################################

import os, time, ffmpy
from natsort import natsorted
import pandas as pd
import soundfile as sf
from tqdm import tqdm


################################################################################
##                            HELPER FUNCTIONS                                ##
################################################################################

# function to clean labels
def convertlabels(sortlist, labels, textlabels):
    clabels = list()
    try:
        index = labels.index(sortlist)
        clabel = textlabels[index]
        # pull out converted label
        clabels.append(clabel)
    except:
        clabels = []

    return clabels


def download_audio(link):
    listdir = os.listdir()
    # TODO os.system("youtube-dl -f 'bestaudio[ext=m4a]' ytsearch:'%s'" % (link))
    os.system("youtube-dl --format m4a %s" % (link))
    listdir2 = os.listdir()
    filename = ''
    for i in range(len(listdir2)):
        if listdir2[i] not in listdir and listdir2[i].endswith('.m4a'):
            filename = listdir2[i]
            break

    return filename


################################################################################
##                            MAIN SCRIPT                                     ##
################################################################################
# TODO
d_sound = ['/m/01b_21', '/m/05tny_', '/m/0ghcn6', '/m/0cdnk', '/m/078jl', '/t/dd00036', '/t/dd00037',
           '/m/02_41', '/m/0k4j', '/m/01bjv', '/m/04qvtq', '/m/012n7d', '/m/012ndj', '/m/0284vy3',
           '/m/02x984l', '/m/03l9g', '/m/01d380', '/m/0g6b5', '/m/07qnq_y', '/m/07pws3f', '/m/07pjjrj',
           '/m/07pc8lb', '/m/04_sv', '/m/07qb_dv', '/m/07qv4k0', '/m/07plct2']

s_sound = ['/t/dd00129', '/m/015p6', '/m/0ytgt', '/m/01h8n0', '/m/01j3sz', '/t/dd00001', '/m/03m9d0z',
           '/m/07pggtn', '/m/07pbtc8',  '/m/0838f', '/m/0j6m2', '/m/04k94']

defaultdir = os.getcwd()
os.chdir(defaultdir)

# load labels of the videos

# number, label, words
loadfile = pd.read_excel('labels.xlsx')

number = loadfile.iloc[:, 0].tolist()
labels = loadfile.iloc[:, 1].tolist()
textlabels = loadfile.iloc[:, 2].tolist()

# remove spaces for folders
for i in range(len(textlabels)):
    textlabels[i] = textlabels[i].replace(' ', '')

# now load data for youtube
loadfile2 = pd.read_excel('unbalanced_train_segments.xlsx')

# ylabels have to be cleaned to make a good list (CSV --> LIST)
yid = loadfile2.iloc[:, 0].tolist()[2:]
ystart = loadfile2.iloc[:, 1].tolist()[2:]
yend = loadfile2.iloc[:, 2].tolist()[2:]
ylabels = loadfile2.iloc[:, 3].tolist()[2:]

print(set(ylabels))

# make folders
try:
    defaultdir2 = os.getcwd() + '\\audiosetdata\\'
    os.chdir(os.getcwd() + '\\audiosetdata')
except:
    defaultdir2 = os.getcwd() + '\\audiosetdata\\'
    os.mkdir(os.getcwd() + '\\audiosetdata')
    os.chdir(os.getcwd() + '\\audiosetdata')

existing_wavfiles = list()
for i in range(len(textlabels)):
    if d_sound.count(labels[i]) == 0:  # TODO
        continue

    try:
        os.mkdir(textlabels[i])
    except:
        os.chdir(textlabels[i])
        listdir = os.listdir()
        for j in range(len(listdir)):
            if listdir[j].endswith('.wav'):
                existing_wavfiles.append(listdir[j])
        os.chdir(defaultdir2)

# get last file checkpoint to leave off
existing_wavfiles = natsorted(existing_wavfiles)
print(existing_wavfiles)
try:
    lastfile = int(existing_wavfiles[-1][7:][0:-4])
except:
    lastfile = 0

# iterate through entire CSV file, look for '--' if found, find index, delete section, then go to next index

slink = "https://www.youtube.com/watch?v="

for i in tqdm(range(len(yid))):
    if i < lastfile:
        print('skipping, already downloaded file...')
    else:
        link = slink + yid[i]
        start = float(ystart[i])
        end = float(yend[i])
        print(ylabels[i])
        clabels = convertlabels(ylabels[i], labels, textlabels)
        print(clabels)

        if clabels != [] and d_sound.count(ylabels[i]) > 0:  # TODO

            # change to the right directory
            newdir = defaultdir2 + clabels[0] + '\\'
            print('newdir', newdir)
            os.chdir(newdir)
            # if it is the first download, pursue this path to download video
            lastdir = os.getcwd() + '\\'
            print('lastdir', lastdir)

            if 'snipped' + str(i) + '.wav' not in os.listdir():
                print('str(i)', str(i))
                try:
                    # use YouTube DL to download audio
                    filename = download_audio(link)
                    print('link', link)
                    extension = '.m4a'
                    print('extension', extension)
                    # get file extension and convert to .wav for processing later
                    # TODO error 부분
                    os.rename(filename, '%s%s' % (str(i), extension))
                    print('str(i)',str(i))
                    filename = '%s%s' % (str(i), extension)
                    print('filename', filename)
                    if extension not in ['.wav']:
                        xindex = filename.find(extension)
                        filename = filename[0:xindex]
                        ff = ffmpy.FFmpeg(
                            inputs={filename + extension: None},
                            outputs={filename + '.wav': None})
                        ff.run()
                        # TODO
                        # song = AudioSegment.from_file(filename, extension)
                        # wav_filename = filename.replace(extension, 'wav')
                        # song.export(filename + extension, format="wav")
                        os.remove(filename + extension)
                    file = filename + '.wav'
                    print('file', file)
                    data, samplerate = sf.read(file)
                    totalframes = len(data)
                    print('totalframes',totalframes)
                    totalseconds = totalframes / samplerate
                    print('totalseconds',totalseconds)
                    startsec = start
                    startframe = samplerate * startsec
                    endsec = end
                    endframe = samplerate * endsec
                    print('startframe',startframe)
                    print('endframe',endframe)
                    sf.write('snipped' + file, data[int(startframe):int(endframe)], samplerate)
                    snippedfile = 'snipped' + file
                    os.remove(file)
                except:
                    print('no urls')
                # sleep 3 second sleep to prevent IP from getting banned
                time.sleep(3)
            else:
                print('skipping, already downloaded file...')