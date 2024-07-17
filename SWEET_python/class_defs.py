from pydantic import BaseModel
from typing import List, Dict, Any, Optional
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


class WasteGeneratedDF(BaseModel):
    df: pd.DataFrame

    class Config:
        arbitrary_types_allowed = True

    @classmethod
    def create(cls, waste_mass: float, waste_fractions: WasteFractions, start_year: int, end_year: int, year_of_data_pop: int, growth_rate_historic: float, growth_rate_future: float):
        waste_types = waste_fractions.model_dump().keys()
        years = list(range(start_year, end_year + 1))
        data = {waste: [] for waste in waste_types}

        for year in years:
            t = year - year_of_data_pop
            growth_rate = growth_rate_historic if year < year_of_data_pop else growth_rate_future
            for waste in waste_types:
                value = waste_mass * getattr(waste_fractions, waste) * (growth_rate ** t)
                data[waste].append(value)

        df = pd.DataFrame(data, index=years)
        return cls(df=df)

class DivsDF(BaseModel):
    # baseline_compost: pd.DataFrame
    # baseline_anaerobic: pd.DataFrame
    # baseline_combustion: pd.DataFrame
    # baseline_recycling: pd.DataFrame
    compost: pd.DataFrame
    anaerobic: pd.DataFrame
    combustion: pd.DataFrame
    recycling: pd.DataFrame

    class Config:
        arbitrary_types_allowed = True

    @classmethod
    def create(cls, divs: DivMasses, start_year: int, end_year: int, year_of_data_pop: int, growth_rate_historic: float, growth_rate_future: float):
        compost = cls._create_dataframe(divs.compost, start_year, end_year, year_of_data_pop, growth_rate_historic, growth_rate_future)
        anaerobic = cls._create_dataframe(divs.anaerobic, start_year, end_year, year_of_data_pop, growth_rate_historic, growth_rate_future)
        combustion = cls._create_dataframe(divs.combustion, start_year, end_year, year_of_data_pop, growth_rate_historic, growth_rate_future)
        recycling = cls._create_dataframe(divs.recycling, start_year, end_year, year_of_data_pop, growth_rate_historic, growth_rate_future)

        return cls(
            # baseline_compost=compost, 
            # baseline_anaerobic=anaerobic, 
            # baseline_combustion=combustion, 
            # baseline_recycling=recycling, 
            compost=compost, 
            anaerobic=anaerobic, 
            combustion=combustion, 
            recycling=recycling
        )

    @staticmethod
    def _create_dataframe(div_masses: WasteMasses, start_year: int, end_year: int, year_of_data_pop: int, growth_rate_historic: float, growth_rate_future: float) -> pd.DataFrame:
        waste_types = div_masses.model_dump().keys()
        years = list(range(start_year, end_year + 1))
        data = {waste: [] for waste in waste_types}

        for year in years:
            t = year - year_of_data_pop
            growth_rate = growth_rate_historic if year < year_of_data_pop else growth_rate_future
            for waste in waste_types:
                value = getattr(div_masses, waste) * (growth_rate ** t)
                data[waste].append(value)

        df = pd.DataFrame(data, index=years)
        return df
    
    def _dst_implement(self, implement_year: int, scenario_div_masses: DivMasses, baseline_div_masses: DivMasses, start_year: int, end_year: int, year_of_data_pop: int, growth_rate_historic: float, growth_rate_future: float, components: List[str]) -> None:
        def _get_value_for_year(div_masses: WasteMasses, waste: str, year: int, year_of_data_pop: int, growth_rate_historic: float, growth_rate_future: float) -> float:
            t = year - year_of_data_pop
            growth_rate = growth_rate_historic if year < year_of_data_pop else growth_rate_future
            return getattr(div_masses, waste) * (growth_rate ** t)

        def _create_dynamic_dataframe(start_year: int, end_year: int, year_of_data_pop: int, baseline_div_masses: WasteMasses, scenario_div_masses: WasteMasses, implement_year: int, growth_rate_historic: float, growth_rate_future: float, components: List[str]) -> pd.DataFrame:
            waste_types = components
            years = list(range(start_year, end_year + 1))
            data = {waste: [] for waste in waste_types}

            for year in years:
                for waste in waste_types:
                    if year < implement_year:
                        value = _get_value_for_year(baseline_div_masses, waste, year, year_of_data_pop, growth_rate_historic, growth_rate_future)
                    else:
                        value = _get_value_for_year(scenario_div_masses, waste, year, year_of_data_pop, growth_rate_historic, growth_rate_future)
                    data[waste].append(value)

            df = pd.DataFrame(data, index=years)
            return df

        self.compost = _create_dynamic_dataframe(start_year, end_year, year_of_data_pop, baseline_div_masses.compost, scenario_div_masses.compost, implement_year, growth_rate_historic, growth_rate_future, components)
        self.anaerobic = _create_dynamic_dataframe(start_year, end_year, year_of_data_pop, baseline_div_masses.anaerobic, scenario_div_masses.anaerobic, implement_year, growth_rate_historic, growth_rate_future, components)
        self.combustion = _create_dynamic_dataframe(start_year, end_year, year_of_data_pop, baseline_div_masses.combustion, scenario_div_masses.combustion, implement_year, growth_rate_historic, growth_rate_future, components)
        self.recycling = _create_dynamic_dataframe(start_year, end_year, year_of_data_pop, baseline_div_masses.recycling, scenario_div_masses.recycling, implement_year, growth_rate_historic, growth_rate_future, components)

        return

class LandfillWasteMassDF(BaseModel):
    df: pd.DataFrame

    class Config:
        arbitrary_types_allowed = True

    @classmethod
    # THIS CAN FOR SURE BE MADE FASTER WIThOUT LOOPS?
    def create(cls, waste_generated_df: pd.DataFrame, divs_df: DivsDF, fraction_of_waste: float, waste_types: List[str]) -> None:
        years = waste_generated_df.index
        data = {waste: [] for waste in waste_types}

        for year in years:
            for waste in waste_types:
                total_waste = waste_generated_df.loc[year, waste]
                try: # This is ugly, come back and clean up maybe...the problem is cityparams model_dump, which is built in.
                    diverted_waste = (
                        divs_df.compost.loc[year, waste] + 
                        divs_df.anaerobic.loc[year, waste] + 
                        divs_df.combustion.loc[year, waste] + 
                        divs_df.recycling.loc[year, waste]
                    )
                except:
                    diverted_waste = (
                        divs_df['compost'].loc[year, waste] + 
                        divs_df['anaerobic'].loc[year, waste] + 
                        divs_df['combustion'].loc[year, waste] + 
                        divs_df['recycling'].loc[year, waste]
                    )
                landfill_waste = (total_waste - diverted_waste) * fraction_of_waste
                data[waste].append(landfill_waste)

        df = pd.DataFrame(data, index=years)
        return cls(df=df)