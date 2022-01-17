# MRL Documentation
Short description of lthe language

## Getting started
Lets take a look in the MRL language by creating a simple MRL report for a Polisonomgraphy study (PSG) as follows.

1. Create a file with the extension ```.mrl``` 
2. add the following code. Make sure you have the file name correctly linked to the report in order to access the exam results within your report.
3. Compile the report. 

```yaml
from utils import Diagnosis;
Identification:
    patientName: 'Patient Name';
    Birthdate: '01-01-2000';

Recording:
    name: 'exam.edf'
    type: 'PSG'

Results:
    Respiration: IAH, ODI;

Conclusions:
    Diagnosis: Diagnosis.SAOS.severe;
    Recomendatios: CPAP;
```
The result will be a beaultiful, objective and uniformized PSG report.
## MRL Basics
### Objects
The concept of objects is usefull for programmers who are integrating the language into a system. 
A healthcare professional writing a report doesn't need worry about the conecpt of objects but will be able to take advantage of this abstraction while writing reports.

Objects are represented in a simple ```key:value``` form. E.g.
```yaml
Person:
    name: 'Patient Name';
    Birthdate: '01-01-2000';
```

The propeties of objects can be accessed with dot.
```yaml
Person.name
```

The reserved word ```this``` refers to the current object while constructing an object.. E.g.

```yaml

Person:
    name: 'Patient Name';
    birthdate: '01-01-2000';
    age: getAge(this.Birthdate);
```
### Functions
MRL doenst support general purpose programming but allows the user to import functions implemented in other languages. 

MRL ships with Python integration.
Python was chosen to be the default language because it has great libraries for signal processing and is easy to learn.

In summary: functions must be implemented in Python and then connected to MRL.

In the following example we implemented a function ```getAge(date)``` and connect we connect it to MRL.

```javascript
function getAge(date) -> call python_module.get_age date;
```

### Libraries
To integrate the language.

### Creating a report
A MRL report has at least 4 mandatory blocks: Identification, RecordingInfo, Results, Conclusion.

A report must start with an identification block, which can be created as follows:

```yaml
Identification:
    patientName: 'Patient Name';
    Birthdate: '08-06-1992';
    Age: age(this.Birthdate);
    ClinicalHistory: 
        Summary: 'Fumador, hipersonolência diurna';
        PreviousReport: PreviousReport(this);

```