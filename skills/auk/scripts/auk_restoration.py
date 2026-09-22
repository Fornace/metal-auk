"""Quality-raising instruction library.

Copied verbatim from `template_pools/restoration/improve_quality` in the Prompt
Enhancer config. Two families: bandwidth extension recovers high frequencies that
were cut, effect removal strips a recording coloration. Each has a separate
instruction per optional cleanup, and a near miss on wording makes the model return
the source audio, so these strings are not paraphrased.
"""

from __future__ import annotations

# Quality raising. Upstream splits this into two families and tunes a separate
# instruction for every combination of the task and an optional cleanup. The strings
# below are copied verbatim from pe.config.yaml `template_pools/restoration/improve_quality`,
# because a near miss on wording is a silent no-op: the model returns the source audio.
BANDWIDTH = {
    "": "This audio suffers from limited bandwidth. Please restore it to a wideband, clear-sounding speech.",
    "denoise": (
        "Please perform bandwidth extension on this speech to recover the missing high-frequency "
        "content, while removing the background noise, outputting a wideband clean speech."
    ),
    "dereverb": (
        "Please perform bandwidth extension on this speech to recover the missing high-frequency "
        "content, while removing the room reverberation, outputting a wideband clean speech."
    ),
    "denoise_dereverb": (
        "Please perform bandwidth extension on this speech to recover the missing high-frequency "
        "content, while removing the background noise and room reverberation, outputting a wideband "
        "clean speech."
    ),
}

RESTORE = {
    "telephone": {
        "": "This speech carries telephone-band coloration. Please restore it to a natural wideband voice.",
        "denoise": (
            "Please remove the telephone-band coloration from this audio while removing the background "
            "noise, and output a normally-bandwidthed clean voice."
        ),
        "dereverb": (
            "Please remove the telephone-band coloration and room reverberation from this audio, and "
            "output a normally-bandwidthed clean voice."
        ),
        "denoise_dereverb": (
            "Please remove the telephone-band coloration from this audio while removing the background "
            "noise and room reverberation, and output a normally-bandwidthed clean voice."
        ),
    },
    "megaphone": {
        "": "This audio has a megaphone-like coloration. Please restore it to a natural-sounding voice.",
        "denoise": (
            "Please remove the megaphone coloration from this audio while removing the background "
            "noise, and output a natural, clear voice."
        ),
        "dereverb": (
            "Please remove the megaphone coloration and room reverberation from this audio, and output "
            "a natural, clear voice."
        ),
        "denoise_dereverb": (
            "Please remove the megaphone coloration from this audio while removing the background noise "
            "and room reverberation, and output a natural, clear voice."
        ),
    },
    "underwater": {
        "": "This audio sounds underwater / muffled. Please restore it to a normal, clear-sounding voice.",
        "denoise": (
            "Please remove the underwater / muffled coloration and background noise from this audio, "
            "outputting a clear wideband voice."
        ),
        "dereverb": (
            "Please remove the underwater / muffled coloration and room reverberation from this audio, "
            "outputting a clear wideband voice."
        ),
        "denoise_dereverb": (
            "Please remove the underwater / muffled coloration from this audio while removing the "
            "background noise and room reverberation, outputting a clear wideband voice."
        ),
    },
    "clipping": {
        "": "This audio is clipped. Please declip it and restore the speech to a natural-looking waveform.",
        "denoise": (
            "Please repair the hard-clipped waveform in this speech while removing the background noise, "
            "producing a clean and intact voice."
        ),
        "dereverb": (
            "Please repair the hard-clipped waveform in this speech while removing the room "
            "reverberation, producing a clean and intact voice."
        ),
        "denoise_dereverb": (
            "Please repair the hard-clipped waveform in this speech while removing the background noise "
            "and room reverberation, producing a clean and intact voice."
        ),
    },
    "dropout": {
        "": (
            "This audio has audible packet dropouts or short cut-offs. Please fill in the missing "
            "segments, outputting a continuous, intact voice."
        ),
        "denoise": (
            "Please repair the packet-dropout artifacts in this speech while removing the background "
            "noise, and output a continuous, natural voice."
        ),
        "dereverb": (
            "Please repair the packet-dropout artifacts in this speech while removing the room "
            "reverberation, and output a continuous, natural voice."
        ),
        "denoise_dereverb": (
            "Please repair the packet-dropout artifacts in this speech while removing the background "
            "noise and room reverberation, and output a continuous, natural voice."
        ),
    },
    "dc_offset": {
        "": "This audio has a DC offset. Please remove the DC component and output a properly-centered clean voice.",
        "denoise": (
            "This audio has a DC offset. Please remove the DC component and background noise, and "
            "output a properly-centered clean voice."
        ),
        "dereverb": (
            "This audio has a DC offset. Please remove the DC component and room reverberation, and "
            "output a properly-centered clean voice."
        ),
        "denoise_dereverb": (
            "This audio has a DC offset. Please remove the DC component, background noise, and room "
            "reverberation, and output a properly-centered clean voice."
        ),
    },
    "_generic": {
        "": (
            "Please restore the audio quality by removing the recording/coloration artifacts, and "
            "output a natural, clear-sounding wideband voice."
        ),
    },
}

