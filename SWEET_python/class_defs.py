
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Union, TypeVar, Generic, Tuple, Annotated
import pandas as pd
import numpy as np
from enum import Enum

class WasteFractions(BaseModel):
    food: float = 0.0
    green: float = 0.0
    wood: float = 0.0
    paper_cardboard: float = 0.0
    textiles: float = 0.0
    plastic: float = 0.0
    metal: float = 0.0
    glass: float = 0.0
    rubber: float = 0.0
    other: float = 0.0

# class WasteFractions(BaseModel):
#     food: pd.Series
#     green: pd.Series
#     wood: pd.Series
#     paper_cardboard: pd.Series
#     textiles: pd.Series
#     plastic: pd.Series
#     metal: pd.Series
#     glass: pd.Series
#     rubber: pd.Series
#     other: pd.Series

#     class Config:
#         arbitrary_types_allowed = True

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
    combustion: Union[float, pd.Series]
    recycling: float

    class Config:
        arbitrary_types_allowed = True

class DivComponentFractions(BaseModel):
    compost: WasteFractions
    anaerobic: WasteFractions
    combustion: WasteFractions
    recycling: WasteFractions

class DivComponentFractionsDF(BaseModel):
    compost: pd.DataFrame
    anaerobic: pd.DataFrame
    combustion: pd.DataFrame
    recycling: pd.DataFrame

    class Config:
        arbitrary_types_allowed = True

class DivMasses(BaseModel):
    compost: WasteMasses
    anaerobic: WasteMasses
    combustion: Union[WasteMasses, pd.DataFrame]
    recycling: WasteMasses

    class Config:
        arbitrary_types_allowed = True

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

# class DecompositionRates(BaseModel):
#     food: float
#     green: float
#     wood: float
#     paper_cardboard: float
#     textiles: float

class DecompositionRates(BaseModel):
    food: pd.Series
    green: pd.Series
    wood: pd.Series
    paper_cardboard: pd.Series
    textiles: pd.Series

    class Config:
        arbitrary_types_allowed = True

class WasteGeneratedDF(BaseModel):
    df: pd.DataFrame

    class Config:
        arbitrary_types_allowed = True

    # This should just take waste_masses as an input instead?
    # @classmethod
    # def create(cls, waste_mass: float, waste_fractions: WasteFractions, start_year: int, end_year: int, year_of_data_pop: int, growth_rate_historic: float, growth_rate_future: float):
    #     waste_types = waste_fractions.model_dump().keys()
    #     years = list(range(start_year, end_year + 1))
    #     data = {waste: [] for waste in waste_types}

    #     for year in years:
    #         t = year - year_of_data_pop
    #         growth_rate = growth_rate_historic if year < year_of_data_pop else growth_rate_future
    #         for waste in waste_types:
    #             value = waste_mass * getattr(waste_fractions, waste) * (growth_rate ** t)
    #             data[waste].append(value)

    #     df = pd.DataFrame(data, index=years)
    #     return cls(df=df)

    @classmethod
    def create(cls, waste_masses_df: pd.DataFrame, start_year: int, end_year: int, year_of_data_pop: int, growth_rate_historic: float, growth_rate_future: float):
        years = np.arange(start_year, end_year + 1)
        t = years - year_of_data_pop

        # Create growth rate array, using growth_rate_historic for years before year_of_data_pop and growth_rate_future after
        growth_rate = np.where(years < year_of_data_pop, growth_rate_historic, growth_rate_future)
        growth_factors = growth_rate ** t

        # Apply growth factors to the waste_masses_df
        adjusted_data = waste_masses_df.multiply(growth_factors, axis=0)

        return cls(df=adjusted_data)
    
    @classmethod
    def create_advanced(cls, waste_masses_df: pd.DataFrame, start_year: int, end_year: int, year_of_data_pop: int, growth_rate_historic: float, growth_rate_future: float):
        years = np.arange(start_year, end_year + 1)
        t = years - year_of_data_pop

        # Create growth rate array, using growth_rate_historic for years before year_of_data_pop and growth_rate_future after
        growth_rate = np.where(years < year_of_data_pop, growth_rate_historic, growth_rate_future)
        growth_factors = growth_rate ** t

        # Apply growth factors to each row of the DataFrame
        adjusted_data = waste_masses_df.multiply(growth_factors, axis=0)

        return cls(df=adjusted_data)
    
    def dst_implement_advanced(self, waste_masses: pd.DataFrame, implement_year: int, new_waste_mass: float, new_waste_fractions: WasteFractions, year_of_data_pop: int, growth_rate_historic: float, growth_rate_future: float):
        # Ensure implement_year is within the DataFrame's index range
        if implement_year < waste_masses.index.min() or implement_year > waste_masses.index.max():
            raise ValueError(f"Implement year {implement_year} is out of the DataFrame's index range.")

        # Create new data starting from the implement_year
        waste_types = new_waste_fractions.model_dump().keys()
        years = list(range(implement_year, waste_masses.index.max() + 1))
        new_data = {waste: [] for waste in waste_types}

        for year in years:
            t = year - year_of_data_pop
            growth_rate = growth_rate_historic if year < year_of_data_pop else growth_rate_future
            for waste in waste_types:
                value = new_waste_mass * getattr(new_waste_fractions, waste) * (growth_rate ** t)
                new_data[waste].append(value)

        # Create a DataFrame for new data
        new_df = pd.DataFrame(new_data, index=years)

        # Update the original DataFrame
        updated_df = waste_masses.copy()
        updated_df.loc[implement_year:] = new_df

        self.df = updated_df

class DivsDF(BaseModel):
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
    
    def sum(self) -> pd.DataFrame:
        all_columns = self.compost.columns.union(self.anaerobic.columns).union(self.combustion.columns).union(self.recycling.columns)
        
        # Reindex and fill missing values, then infer object types
        divs_list = [
            self.compost.reindex(columns=all_columns).infer_objects(copy=False).fillna(0),
            self.anaerobic.reindex(columns=all_columns).infer_objects(copy=False).fillna(0),
            self.combustion.reindex(columns=all_columns).infer_objects(copy=False).fillna(0),
            self.recycling.reindex(columns=all_columns).infer_objects(copy=False).fillna(0)
        ]
        
        return sum(divs_list)
    
    # THIS NEEDS TO ACCOMODATE NEW_BASELINE, and i think can be rewritten to use df, be more efficient. same with the one below. 
    def _dst_implement(
            self, 
            implement_year: int, 
            scenario_div_masses: DivMasses, 
            baseline_div_masses: DivMasses, 
            start_year: int, 
            end_year: int, 
            year_of_data_pop: int, 
            growth_rate_historic: float, 
            growth_rate_future: float, 
            components: List[str]
        ) -> None:
        
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

        self.compost = _create_dynamic_dataframe(start_year, end_year, year_of_data_pop, baseline_div_masses.compost, scenario_div_masses.compost, implement_year, growth_rate_historic, growth_rate_future, components['compost'])
        self.anaerobic = _create_dynamic_dataframe(start_year, end_year, year_of_data_pop, baseline_div_masses.anaerobic, scenario_div_masses.anaerobic, implement_year, growth_rate_historic, growth_rate_future, components['anaerobic'])
        self.combustion = _create_dynamic_dataframe(start_year, end_year, year_of_data_pop, baseline_div_masses.combustion, scenario_div_masses.combustion, implement_year, growth_rate_historic, growth_rate_future, components['combustion'])
        self.recycling = _create_dynamic_dataframe(start_year, end_year, year_of_data_pop, baseline_div_masses.recycling, scenario_div_masses.recycling, implement_year, growth_rate_historic, growth_rate_future, components['recycling'])

        return
    
    # This sucks now...or at least still has dumb loop
    def _dst_implement_advanced(
        self, 
        implement_year: int, 
        scenario_div_masses: DivMasses, 
        baseline_div_masses: DivMasses, 
        start_year: int, 
        end_year: int, 
        year_of_data_pop: int, 
        growth_rate_historic: float, 
        growth_rate_future: float, 
        components: List[str]
    ) -> None:
        
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
        
        def _create_dynamic_dataframe_combustion(start_year: int, end_year: int, year_of_data_pop: int, baseline_div_masses: WasteMasses, scenario_div_masses: WasteMasses, implement_year: int, growth_rate_historic: float, growth_rate_future: float, components: List[str]) -> pd.DataFrame:

            scenario_div_masses.index = scenario_div_masses.index.astype(int)

            scenario_div_masses['years_from_ref'] = scenario_div_masses.index - year_of_data_pop
            
            # Apply growth rates
            scenario_div_masses.loc[scenario_div_masses.index < year_of_data_pop, 'growth_rate'] = growth_rate_historic
            scenario_div_masses.loc[scenario_div_masses.index >= year_of_data_pop, 'growth_rate'] = growth_rate_future

            # Apply growth to all values in the rows
            for col in scenario_div_masses.columns[:-2]:  # Exclude 'years_from_ref' and 'growth_rate' columns
                scenario_div_masses[col] = scenario_div_masses[col] * (scenario_div_masses['growth_rate'] ** scenario_div_masses['years_from_ref'])

            # Drop the helper columns
            scenario_div_masses.drop(columns=['years_from_ref', 'growth_rate'], inplace=True)

            return scenario_div_masses

        self.compost = _create_dynamic_dataframe(start_year, end_year, year_of_data_pop, baseline_div_masses.compost, scenario_div_masses.compost, implement_year, growth_rate_historic, growth_rate_future, components['compost'])
        self.anaerobic = _create_dynamic_dataframe(start_year, end_year, year_of_data_pop, baseline_div_masses.anaerobic, scenario_div_masses.anaerobic, implement_year, growth_rate_historic, growth_rate_future, components['anaerobic'])
        self.combustion = _create_dynamic_dataframe_combustion(start_year, end_year, year_of_data_pop, baseline_div_masses.combustion, scenario_div_masses.combustion, implement_year, growth_rate_historic, growth_rate_future, components['combustion'])
        self.recycling = _create_dynamic_dataframe(start_year, end_year, year_of_data_pop, baseline_div_masses.recycling, scenario_div_masses.recycling, implement_year, growth_rate_historic, growth_rate_future, components['recycling'])

        return
    
    @classmethod
    def create_advanced_baseline(cls, divs: DivMasses, year_of_data_pop: int, growth_rate_historic: float, growth_rate_future: float):
        compost = cls._apply_growth_rate(divs.compost, year_of_data_pop, growth_rate_historic, growth_rate_future)
        anaerobic = cls._apply_growth_rate(divs.anaerobic, year_of_data_pop, growth_rate_historic, growth_rate_future)
        combustion = cls._apply_growth_rate(divs.combustion, year_of_data_pop, growth_rate_historic, growth_rate_future)
        recycling = cls._apply_growth_rate(divs.recycling, year_of_data_pop, growth_rate_historic, growth_rate_future)

        return cls(
            compost=compost, 
            anaerobic=anaerobic, 
            combustion=combustion, 
            recycling=recycling
        )

    @staticmethod
    def _apply_growth_rate(df: pd.DataFrame, year_of_data_pop: int, growth_rate_historic: float, growth_rate_future: float) -> pd.DataFrame:
        years = df.index
        t = years - year_of_data_pop

        # Vectorized growth rate calculation
        growth_rates = np.where(years < year_of_data_pop, growth_rate_historic, growth_rate_future)
        growth_factors = np.power(growth_rates, t)

        # Apply growth factors across all waste types (columns) at once
        adjusted_data = df.multiply(growth_factors, axis=0)

        return adjusted_data
    
    # def _advanced_baseline(
    #     self, 
    #     scenario_div_masses: DivMasses, 
    #     baseline_div_masses: DivMasses, 
    #     start_year: int, 
    #     end_year: int, 
    #     year_of_data_pop: int, 
    #     growth_rate_historic: float, 
    #     growth_rate_future: float, 
    #     components: List[str]
    # ) -> None:
        
    #     def _get_value_for_year(div_masses: WasteMasses, waste: str, year: int, year_of_data_pop: int, growth_rate_historic: float, growth_rate_future: float) -> float:
    #         t = year - year_of_data_pop
    #         growth_rate = growth_rate_historic if year < year_of_data_pop else growth_rate_future
    #         return getattr(div_masses, waste) * (growth_rate ** t)

    #     def _create_dynamic_dataframe(start_year: int, end_year: int, year_of_data_pop: int, baseline_div_masses: WasteMasses, scenario_div_masses: WasteMasses, growth_rate_historic: float, growth_rate_future: float, components: List[str]) -> pd.DataFrame:
    #         waste_types = components
    #         years = list(range(start_year, end_year + 1))
    #         data = {waste: [] for waste in waste_types}

    #         # This loop can definitely be disposed of
    #         for year in years:
    #             for waste in waste_types:
    #                 # if year < implement_year:
    #                 #     value = _get_value_for_year(baseline_div_masses, waste, year, year_of_data_pop, growth_rate_historic, growth_rate_future)
    #                 # else:
    #                 value = _get_value_for_year(scenario_div_masses, waste, year, year_of_data_pop, growth_rate_historic, growth_rate_future)
    #                 data[waste].append(value)

    #         df = pd.DataFrame(data, index=years)
    #         return df
        
    #     def _create_dynamic_dataframe_combustion(start_year: int, end_year: int, year_of_data_pop: int, baseline_div_masses: WasteMasses, scenario_div_masses: WasteMasses, implement_year: int, growth_rate_historic: float, growth_rate_future: float, components: List[str]) -> pd.DataFrame:

    #         scenario_div_masses.index = scenario_div_masses.index.astype(int)

    #         scenario_div_masses['years_from_ref'] = scenario_div_masses.index - year_of_data_pop
            
    #         # Apply growth rates
    #         scenario_div_masses.loc[scenario_div_masses.index < year_of_data_pop, 'growth_rate'] = growth_rate_historic
    #         scenario_div_masses.loc[scenario_div_masses.index >= year_of_data_pop, 'growth_rate'] = growth_rate_future

    #         # Apply growth to all values in the rows
    #         for col in scenario_div_masses.columns[:-2]:  # Exclude 'years_from_ref' and 'growth_rate' columns
    #             scenario_div_masses[col] = scenario_div_masses[col] * (scenario_div_masses['growth_rate'] ** scenario_div_masses['years_from_ref'])

    #         # Drop the helper columns
    #         scenario_div_masses.drop(columns=['years_from_ref', 'growth_rate'], inplace=True)

    #         return scenario_div_masses

    #     self.compost = _create_dynamic_dataframe(start_year, end_year, year_of_data_pop, baseline_div_masses.compost, scenario_div_masses.compost, growth_rate_historic, growth_rate_future, components['compost'])
    #     self.anaerobic = _create_dynamic_dataframe(start_year, end_year, year_of_data_pop, baseline_div_masses.anaerobic, scenario_div_masses.anaerobic, growth_rate_historic, growth_rate_future, components['anaerobic'])
    #     self.combustion = _create_dynamic_dataframe_combustion(start_year, end_year, year_of_data_pop, baseline_div_masses.combustion, scenario_div_masses.combustion, growth_rate_historic, growth_rate_future, components['combustion'])
    #     self.recycling = _create_dynamic_dataframe(start_year, end_year, year_of_data_pop, baseline_div_masses.recycling, scenario_div_masses.recycling, growth_rate_historic, growth_rate_future, components['recycling'])

    #     return

class LandfillWasteMassDF(BaseModel):
    df: pd.DataFrame

    class Config:
        arbitrary_types_allowed = True

    @classmethod
    def create(cls, waste_generated_df: pd.DataFrame, divs_df: DivsDF, fraction_of_waste: float, waste_types: List[str]) -> None:
        # years = waste_generated_df.index
        # data = {waste: [] for waste in waste_types}

        # for year in years:
        #     for waste in waste_types:
        #         total_waste = waste_generated_df.loc[year, waste]
        #         try:
        #             diverted_waste = (
        #                 divs_df.compost.loc[year, waste] + 
        #                 divs_df.anaerobic.loc[year, waste] + 
        #                 divs_df.combustion.loc[year, waste] + 
        #                 divs_df.recycling.loc[year, waste]
        #             )
        #         except:
        #             diverted_waste = (
        #                 divs_df['compost'].loc[year, waste] + 
        #                 divs_df['anaerobic'].loc[year, waste] + 
        #                 divs_df['combustion'].loc[year, waste] + 
        #                 divs_df['recycling'].loc[year, waste]
        #             )
        #         landfill_waste = (total_waste - diverted_waste) * fraction_of_waste
        #         data[waste].append(landfill_waste)

        try:
            diverted = divs_df.sum()
        except:
            diverted = sum(divs_df.values())
        
        landfill_waste = (waste_generated_df - diverted) * fraction_of_waste

        #df = pd.DataFrame(data, index=years)
        #return cls(df=df)
        return cls(df=landfill_waste)
    
    @classmethod
    def create_advanced(cls, waste_generated_df: pd.DataFrame, divs_df: DivsDF, fraction_of_waste_series: pd.Series) -> 'LandfillWasteMassDF':
        try:
            diverted = divs_df.sum()
        except:
            diverted = sum(divs_df.values())
        
        fraction_of_waste_series.index = fraction_of_waste_series.index.astype(int)
        landfill_waste = (waste_generated_df - diverted).mul(fraction_of_waste_series, axis=0)

        return cls(df=landfill_waste)
    
    # This needs an underscore before it i think
    def dst_implement_advanced(self, waste_generated_df: pd.DataFrame, divs_df: 'DivsDF', fraction_of_waste_series: pd.Series) -> 'LandfillWasteMassDF':
        # years = waste_generated_df.index
        # data = {waste: [] for waste in waste_types}

        # for year in years:
        #     fraction_of_waste = fraction_of_waste_series.at[str(year)]
        #     for waste in waste_types:
        #         total_waste = waste_generated_df.at[year, waste]
        #         try:
        #             diverted_waste = (
        #                 divs_df.compost.at[year, waste] + 
        #                 divs_df.anaerobic.at[year, waste] + 
        #                 divs_df.combustion.at[year, waste] + 
        #                 divs_df.recycling.at[year, waste]
        #             )
        #         except KeyError:
        #             diverted_waste = (
        #                 divs_df['compost'].at[year, waste] + 
        #                 divs_df['anaerobic'].at[year, waste] + 
        #                 divs_df['combustion'].at[year, waste] + 
        #                 divs_df['recycling'].at[year, waste]
        #             )
        #         landfill_waste = (total_waste - diverted_waste) * fraction_of_waste
        #         data[waste].append(landfill_waste)

        try:
            diverted = divs_df.sum()
        except:
            diverted = sum(divs_df.values())
        
        fraction_of_waste_series.index = fraction_of_waste_series.index.astype(int)
        landfill_waste = (waste_generated_df - diverted).mul(fraction_of_waste_series, axis=0)

        self.df = landfill_waste
        #self.df = pd.DataFrame(data, index=years)


T = TypeVar("T")

class Variant(BaseModel, Generic[T]):
    baseline: T
    scenario: Optional[T] = None

    def __getitem__(self, item):
        if item == "baseline":
            return self.baseline
        elif item == "scenario":
            return self.scenario
        else:
            raise KeyError(f"Invalid key: {item}")

    def __setitem__(self, key, value):
        if key == "baseline":
            self.baseline = value
        elif key == "scenario":
            self.scenario = value
        else:
            raise KeyError(f"Invalid key: {key}")


Fraction = Annotated[float, Field(strict=True, ge=0, le=1)]


class CoverType(int, Enum):
    soil = 0
    membrane = 1


class LandfillType(int, Enum):
    landfill = 0
    controlledDump = 1
    openDump = 2