# report writing

from utils import Diagnosis;
from api import *;

Identification:
    patientName: 'Caju';
    Birthdate: '08-06-1992';
    Age: age(this.Birthdate);
    ClinicalHistory: 
        Summary: 'Fumador, hipersonolência diurna';
        PreviousReport: PreviousReport(this);

Recording:
    Equipment: 'Nox';
    Channels: EEG, EMG, PSG, ECG;
    StartDateTime: ...;
    EndDatetime: ...;
    Duration: this.StartDateTime - this.EndDatetime;

Results:
    Respiration: IAH, ODI;
    SleepArchitecture: SleepTime, Arousals, SleepPhases;
    MovementDisorders: 
        LegMovements, 
        'Legs vs Arousals' : correlate(LegMovement, Arousals); 

Conclusions:
    Diagnosis: Diagnosis.SAOS.severe;
    Recomendatios: CPAP;

Appends:
    figure: ipnogram;
    Table: ['Arousals',]

