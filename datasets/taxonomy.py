import os

def comet_taxonomy():
    core_areas = ['Physiological-Clinical', 'Life-Impact', 'Mortality', 'Resource-use', 'Adverse-effects']
    COMET_LABELS = {\
        core_areas[0]:{\
            0:'Physiological/clinical'
            },
        core_areas[1]:{\
            25:'Physical_functioning',
            26:'Social_functioning',
            27:'Role_functioning',
            28:'Emotional_functioning/wellbeing',
            29:'Cognitive_functioning',
            30:'Global_quality_of_life',
            31:'Perceived_health_status',
            32:'Delivery_of_care',
            33:'Personal_circumstances'
            },
        core_areas[2]: { \
            1: 'Mortality/survival'
        },
        core_areas[3]:{\
            34:'Economic',
            35:'Hospital',
            36:'Need_for_further_intervention',
            37:'Societal/carer_burden'
            },
        core_areas[4]:{
            38:'Adverse_events/effectsn'
        }
        }
    return core_areas, COMET_LABELS