import { past_speakers } from './past_talks.js';
import { upcoming_speakers } from './upcoming_talks.js';
import { create_speaker_table } from './aux_select_speakers.js'

// Select container
const container_recent = document.getElementById("recent");

create_speaker_table(past_speakers, container_recent, 4)

const container_upcoming = document.getElementById("upcoming");

create_speaker_table(upcoming_speakers, container_upcoming, upcoming_speakers.length)
