import { past_speakers } from './past_talks.js';
import { create_speaker_table } from './aux_select_speakers.js'

// Select container
const container = document.getElementById("past");

create_speaker_table(past_speakers, container, past_speakers.length)
