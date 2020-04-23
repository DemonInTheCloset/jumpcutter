from argparse import ArgumentParser
from math import ceil
from os import mkdir, path, rename
from re import search
from shutil import copyfile, rmtree
from subprocess import call

import numpy as np
from audiotsm import phasevocoder
from audiotsm.io.wav import WavReader, WavWriter
from pytube import YouTube
from scipy.io import wavfile


def download_file(url):
    name = YouTube(url).streams.first().download()
    new_name = name.replace(' ', '_')
    rename(name, new_name)
    return new_name


def get_max_volume(s):
    max_volume = float(np.max(s))
    min_volume = float(np.min(s))
    return max(max_volume, -min_volume)


def copy_frame(input_frame, output_frame):
    src = TEMP_FOLDER + "/frame{:06d}".format(input_frame + 1) + ".jpg"
    dst = TEMP_FOLDER + "/newFrame{:06d}".format(output_frame + 1) + ".jpg"
    if not path.isfile(src):
        return False
    copyfile(src, dst)
    if output_frame % 20 == 19:
        print(str(output_frame + 1) + " time-altered frames saved.")
    return True


def input_to_output_filename(filename):
    dot_index = filename.rfind(".")
    return filename[:dot_index] + "_ALTERED" + filename[dot_index:]


def create_path(s):
    # assert (not os.path.exists(s)), "The filepath "+s+" already exists. Don't want to overwrite it. Aborting."

    try:
        mkdir(s)
    except OSError:
        assert False, "Creation of the directory %s failed. (The TEMP folder may already exist. Delete or rename it, and try again.)"


def delete_path(s):  # Dangerous! Watch out!
    try:
        rmtree(s, ignore_errors=False)
    except OSError:
        print("Deletion of the directory %s failed" % s)
        print(OSError)


def parse_arguments():
    parser = ArgumentParser(
            description='Modifies a video file to play at different speeds when there is sound vs. silence.')

    source = parser.add_mutually_exclusive_group(required=True)

    source.add_argument('--input_file', type=str, help='the video file you want modified')
    source.add_argument('--url', type=str, help='A youtube url to download and process')

    parser.add_argument('--output_file', type=str, default="",
                        help="the output file. (optional. if not included, it'll just modify the input file name)")

    # Cutter Options
    parser.add_argument('--silent_threshold', type=float, default=0.03,
                        help="the volume amount that frames' audio needs to surpass to be consider \"sounded\". It ranges from 0 (silence) to 1 (max volume)")

    parser.add_argument('--sounded_speed', type=float, default=1.00,
                        help="the speed that sounded (spoken) frames should be played at. Typically 1.")

    parser.add_argument('--silent_speed', type=float, default=5.00,
                        help="the speed that silent frames should be played at. 999999 for jumpcutting.")

    parser.add_argument('--frame_margin', type=float, default=1,
                        help="some silent frames adjacent to sounded frames are included to provide context. How many frames on either the side of speech should be included? That's this variable.")

    # Quality Options
    parser.add_argument('--sample_rate', type=float, default=44100, help="sample rate of the input and output videos")

    parser.add_argument('--frame_rate', type=float, default=30,
                        help="frame rate of the input and output videos. optional... I try to find it out myself, but it doesn't always work.")

    parser.add_argument('--frame_quality', type=int, default=3,
                        help="quality of frames to be extracted from input video. 1 is highest, 31 is lowest, 3 is the default.")

    return parser.parse_args()


def main():
    args = parse_arguments()

    frame_rate = args.frame_rate
    sample_rate = args.sample_rate
    silent_threshold = args.silent_threshold
    frame_spread = args.frame_margin
    new_speed = [args.silent_speed, args.sounded_speed]

    if args.url is not None:
        input_file = download_file(args.url)
    else:
        input_file = args.input_file

    frame_quality = args.frame_quality

    assert input_file is not None, "why u put no input file, that dum"

    if len(args.output_file) >= 1:
        output_file = args.output_file
    else:
        output_file = input_to_output_filename(input_file)

    temp_folder = "TEMP"
    audio_fade_envelope_size = 400  # smooth out transition audio by quickly fading in/out (arbitrary magic number whatever)

    create_path(temp_folder)

    command = "ffmpeg -i " + input_file + " -qscale:v " + str(
            frame_quality) + " " + temp_folder + "/frame%06d.jpg -hide_banner"
    call(command, shell=True)

    command = "ffmpeg -i " + input_file + " -ab 160k -ac 2 -ar " + str(
            sample_rate) + " -vn " + temp_folder + "/audio.wav"

    call(command, shell=True)

    command = "ffmpeg -i " + temp_folder + "/input.mp4 2>&1"
    f = open(temp_folder + "/params.txt", "w")
    call(command, shell=True, stdout=f)

    sample_rate, audio_data = wavfile.read(temp_folder + "/audio.wav")
    audio_sample_count = audio_data.shape[0]
    max_audio_volume = get_max_volume(audio_data)

    f = open(temp_folder + "/params.txt", 'r+')
    pre_params = f.read()
    f.close()
    params = pre_params.split('\n')
    for line in params:
        m = search('Stream #.*Video.* ([0-9]*) fps', line)
        if m is not None:
            frame_rate = float(m.group(1))

    samples_per_frame = sample_rate / frame_rate

    audio_frame_count = int(ceil(audio_sample_count / samples_per_frame))

    has_loud_audio = np.zeros(audio_frame_count)

    for i in range(audio_frame_count):
        start = int(i * samples_per_frame)
        end = min(int((i + 1) * samples_per_frame), audio_sample_count)
        audio_chunks = audio_data[start:end]
        max_chunks_volume = float(get_max_volume(audio_chunks)) / max_audio_volume
        if max_chunks_volume >= silent_threshold:
            has_loud_audio[i] = 1

    chunks = [[0, 0, 0]]
    should_include_frame = np.zeros(audio_frame_count)
    for i in range(audio_frame_count):
        start = int(max(0, i - frame_spread))
        end = int(min(audio_frame_count, i + 1 + frame_spread))
        should_include_frame[i] = np.max(has_loud_audio[start:end])
        if i >= 1 and should_include_frame[i] != should_include_frame[i - 1]:  # Did we flip?
            chunks.append([chunks[-1][1], i, should_include_frame[i - 1]])

    chunks.append([chunks[-1][1], audio_frame_count, should_include_frame[i - 1]])
    chunks = chunks[1:]

    output_audio_data = np.zeros((0, audio_data.shape[1]))
    output_pointer = 0

    last_existing_frame = None
    for chunk in chunks:
        audio_chunk = audio_data[int(chunk[0] * samples_per_frame):int(chunk[1] * samples_per_frame)]

        s_file = temp_folder + "/tempStart.wav"
        e_file = temp_folder + "/tempEnd.wav"
        wavfile.write(s_file, sample_rate, audio_chunk)
        with WavReader(s_file) as reader:
            with WavWriter(e_file, reader.channels, reader.samplerate) as writer:
                tsm = phasevocoder(reader.channels, speed=new_speed[int(chunk[2])])
                tsm.run(reader, writer)

        _, altered_audio_data = wavfile.read(e_file)
        length = altered_audio_data.shape[0]
        end_pointer = output_pointer + length
        output_audio_data = np.concatenate((output_audio_data, altered_audio_data / max_audio_volume))

        # output_audio_data[output_pointer:end_pointer] = altered_audio_data/max_audio_volume

        # smooth out transition audio by quickly fading in/out

        if length < audio_fade_envelope_size:
            output_audio_data[output_pointer:end_pointer] = 0  # audio is less than 0.01 sec, let's just remove it.
        else:
            pre_mask = np.arange(audio_fade_envelope_size) / audio_fade_envelope_size
            mask = np.repeat(pre_mask[:, np.newaxis], 2, axis=1)  # make the fade-envelope mask stereo
            output_audio_data[output_pointer:output_pointer + audio_fade_envelope_size] *= mask
            output_audio_data[end_pointer - audio_fade_envelope_size:end_pointer] *= 1 - mask

        start_output_frame = int(ceil(output_pointer / samples_per_frame))
        end_output_frame = int(ceil(end_pointer / samples_per_frame))
        for outputFrame in range(start_output_frame, end_output_frame):
            input_frame = int(chunk[0] + new_speed[int(chunk[2])] * (outputFrame - start_output_frame))
            did_it_work = copy_frame(input_frame, outputFrame)
            if did_it_work:
                last_existing_frame = input_frame
            else:
                copy_frame(last_existing_frame, outputFrame)

        output_pointer = end_pointer

    wavfile.write(temp_folder + "/audioNew.wav", sample_rate, output_audio_data)

    '''
    outputFrame = math.ceil(output_pointer/samples_per_frame)
    for endGap in range(outputFrame,audio_frame_count):
        copyFrame(int(audioSampleCount/samples_per_frame)-1,endGap)
    '''

    command = "ffmpeg -framerate " + str(
            frame_rate) + " -i " + temp_folder + "/newFrame%06d.jpg -i " + temp_folder + "/audioNew.wav -strict -2 " + output_file
    call(command, shell=True)

    delete_path(temp_folder)


if __name__ == '__main__':
    main()
