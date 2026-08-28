import { workshop_data } from './workshop_data.js';
import { create_confirmed_speaker_table, create_timetable } from '/../../../js/build_tables.js'

// add confirmed speakers
const speaker_container = document.getElementById("confirmed");

create_confirmed_speaker_table(workshop_data, speaker_container);


// add information on dates
const container = document.getElementById("day1");
create_timetable(workshop_data, container, "2026-10-19");

const container2 = document.getElementById("day2");
create_timetable(workshop_data, container2, "2026-10-20");

const container3 = document.getElementById("day3");
create_timetable(workshop_data, container3, "2026-10-21");
