from pydantic import BaseModel
import pandas as pd

class WasteFractions(BaseModel):
    food: float
    green: float
    wood: float
    paper_cardboard: float
    textiles: float
    plastic: float
    metal: float
    glass: float
    rubber: float
    other: float

class WasteMasses(BaseModel):
    food: float
    green: float
    wood: float
    paper_cardboard: float
    textiles: float
    plastic: float
    metal: float
    glass: float
    rubber: float
    other: float

class DiversionFractions(BaseModel):
    compost: float
    anaerobic: float
    combustion: float
    recycling: float

class DivComponentFractions(BaseModel):
    compost: WasteFractions
    anaerobic: WasteFractions
    combustion: WasteFractions
    recycling: WasteFractions

class DivMasses(BaseModel):
    compost: WasteMasses
    anaerobic: WasteMasses
    combustion: WasteMasses
    recycling: WasteMasses

class DivMassesAnnual(BaseModel):
    compost: pd.DataFrame
    anaerobic: pd.DataFrame
    combustion: pd.DataFrame
    recycling: pd.DataFrame

    class Config:
        arbitrary_types_allowed = True

    def to_dict(self):
        return {
            'compost': self.compost.to_dict(orient='records'),
            'anaerobic': self.anaerobic.to_dict(orient='records'),
            'combustion': self.combustion.to_dict(orient='records'),
            'recycling': self.recycling.to_dict(orient='records')
        }

class SplitFractions(BaseModel):
    landfill_w_capture: float
    landfill_wo_capture: float
    dumpsite: float

class DecompositionRates(BaseModel):
    food: float
    green: float
    wood: float
    paper_cardboard: float
    textiles: float

