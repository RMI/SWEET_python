"""World Bank income group by country, as Climate TRACE reads it.

Copied from ``raw_emissions.raw_world_bank_country_level`` in the WasteMAP
database on 2026-10-13. That table is loaded, unchanged, from
``country_level_data_0.csv`` in the ``worldbank`` blob container, which is the
country file of the World Bank's What a Waste 2.0 dataset (2018). Its income
groups are the World Bank classification of that time, not the current one:
Argentina is high income here, and Bulgaria and Costa Rica are not.

Keep this in step with that table rather than with the latest World Bank list.
Climate TRACE's pipeline reads the table for every site, and the point of
:mod:`SWEET_python.site_type_defaults` is that every tool reaches the same
answer for the same country.

Countries the table does not list (overseas departments, uninhabited
territories, North Korea, the Holy See) are absent here too, so they have no
income group.
"""

from typing import Dict

#: ``HIC`` high, ``UMC`` upper-middle, ``LMC`` lower-middle, ``LIC`` low.
INCOME_ID_BY_ISO3: Dict[str, str] = {
    "ABW": "HIC",  # Aruba
    "AFG": "LIC",  # Afghanistan
    "AGO": "LMC",  # Angola
    "ALB": "UMC",  # Albania
    "AND": "HIC",  # Andorra
    "ARE": "HIC",  # United Arab Emirates
    "ARG": "HIC",  # Argentina
    "ARM": "UMC",  # Armenia
    "ASM": "UMC",  # American Samoa
    "ATG": "HIC",  # Antigua and Barbuda
    "AUS": "HIC",  # Australia
    "AUT": "HIC",  # Austria
    "AZE": "UMC",  # Azerbaijan
    "BDI": "LIC",  # Burundi
    "BEL": "HIC",  # Belgium
    "BEN": "LIC",  # Benin
    "BFA": "LIC",  # Burkina Faso
    "BGD": "LMC",  # Bangladesh
    "BGR": "UMC",  # Bulgaria
    "BHR": "HIC",  # Bahrain
    "BHS": "HIC",  # Bahamas, The
    "BIH": "UMC",  # Bosnia and Herzegovina
    "BLR": "UMC",  # Belarus
    "BLZ": "UMC",  # Belize
    "BMU": "HIC",  # Bermuda
    "BOL": "LMC",  # Bolivia
    "BRA": "UMC",  # Brazil
    "BRB": "HIC",  # Barbados
    "BRN": "HIC",  # Brunei Darussalam
    "BTN": "LMC",  # Bhutan
    "BWA": "UMC",  # Botswana
    "CAF": "LIC",  # Central African Republic
    "CAN": "HIC",  # Canada
    "CHE": "HIC",  # Switzerland
    "CHI": "HIC",  # Channel Islands
    "CHL": "HIC",  # Chile
    "CHN": "UMC",  # China
    "CIV": "LMC",  # Côte d’Ivoire
    "CMR": "LMC",  # Cameroon
    "COD": "LIC",  # Congo, Dem. Rep.
    "COG": "LMC",  # Congo, Rep.
    "COL": "UMC",  # Colombia
    "COM": "LIC",  # Comoros
    "CPV": "LMC",  # Cabo Verde
    "CRI": "UMC",  # Costa Rica
    "CUB": "UMC",  # Cuba
    "CUW": "HIC",  # Curacao
    "CYM": "HIC",  # Cayman Islands
    "CYP": "HIC",  # Cyprus
    "CZE": "HIC",  # Czech Republic
    "DEU": "HIC",  # Germany
    "DJI": "LMC",  # Djibouti
    "DMA": "UMC",  # Dominica
    "DNK": "HIC",  # Denmark
    "DOM": "UMC",  # Dominican Republic
    "DZA": "UMC",  # Algeria
    "ECU": "UMC",  # Ecuador
    "EGY": "LMC",  # Egypt, Arab Rep.
    "ERI": "LIC",  # Eritrea
    "ESP": "HIC",  # Spain
    "EST": "HIC",  # Estonia
    "ETH": "LIC",  # Ethiopia
    "FIN": "HIC",  # Finland
    "FJI": "UMC",  # Fiji
    "FRA": "HIC",  # France
    "FRO": "HIC",  # Faeroe Islands
    "FSM": "LMC",  # Micronesia, Fed. Sts.
    "GAB": "UMC",  # Gabon
    "GBR": "HIC",  # United Kingdom
    "GEO": "LMC",  # Georgia
    "GHA": "LMC",  # Ghana
    "GIB": "HIC",  # Gibraltar
    "GIN": "LIC",  # Guinea
    "GMB": "LIC",  # Gambia, The
    "GNB": "LIC",  # Guinea-Bissau
    "GNQ": "UMC",  # Equatorial Guinea
    "GRC": "HIC",  # Greece
    "GRD": "UMC",  # Grenada
    "GRL": "HIC",  # Greenland
    "GTM": "UMC",  # Guatemala
    "GUM": "HIC",  # Guam
    "GUY": "UMC",  # Guyana
    "HKG": "HIC",  # Hong Kong SAR, China
    "HND": "LMC",  # Honduras
    "HRV": "HIC",  # Croatia
    "HTI": "LIC",  # Haiti
    "HUN": "HIC",  # Hungary
    "IDN": "LMC",  # Indonesia
    "IMN": "HIC",  # Isle of Man
    "IND": "LMC",  # India
    "IRL": "HIC",  # Ireland
    "IRN": "UMC",  # Iran, Islamic Rep.
    "IRQ": "UMC",  # Iraq
    "ISL": "HIC",  # Iceland
    "ISR": "HIC",  # Israel
    "ITA": "HIC",  # Italy
    "JAM": "UMC",  # Jamaica
    "JOR": "UMC",  # Jordan
    "JPN": "HIC",  # Japan
    "KAZ": "UMC",  # Kazakhstan
    "KEN": "LMC",  # Kenya
    "KGZ": "LMC",  # Kyrgyz Republic
    "KHM": "LMC",  # Cambodia
    "KIR": "LMC",  # Kiribati
    "KNA": "HIC",  # St. Kitts and Nevis
    "KOR": "HIC",  # Korea, Rep.
    "KWT": "HIC",  # Kuwait
    "LAO": "LMC",  # Lao PDR
    "LBN": "UMC",  # Lebanon
    "LBR": "LIC",  # Liberia
    "LBY": "UMC",  # Libya
    "LCA": "UMC",  # St. Lucia
    "LIE": "HIC",  # Liechtenstein
    "LKA": "LMC",  # Sri Lanka
    "LSO": "LMC",  # Lesotho
    "LTU": "HIC",  # Lithuania
    "LUX": "HIC",  # Luxembourg
    "LVA": "HIC",  # Latvia
    "MAC": "HIC",  # Macao SAR, China
    "MAF": "HIC",  # St. Martin (French part)
    "MAR": "LMC",  # Morocco
    "MCO": "HIC",  # Monaco
    "MDA": "LMC",  # Moldova
    "MDG": "LIC",  # Madagascar
    "MDV": "UMC",  # Maldives
    "MEX": "UMC",  # Mexico
    "MHL": "UMC",  # Marshall Islands
    "MKD": "UMC",  # Macedonia, FYR
    "MLI": "LIC",  # Mali
    "MLT": "HIC",  # Malta
    "MMR": "LMC",  # Myanmar
    "MNE": "UMC",  # Montenegro
    "MNG": "LMC",  # Mongolia
    "MNP": "HIC",  # Northern Mariana Islands
    "MOZ": "LIC",  # Mozambique
    "MRT": "LMC",  # Mauritania
    "MUS": "UMC",  # Mauritius
    "MWI": "LIC",  # Malawi
    "MYS": "UMC",  # Malaysia
    "NAM": "UMC",  # Namibia
    "NCL": "HIC",  # New Caledonia
    "NER": "LIC",  # Niger
    "NGA": "LMC",  # Nigeria
    "NIC": "LMC",  # Nicaragua
    "NLD": "HIC",  # Netherlands
    "NOR": "HIC",  # Norway
    "NPL": "LIC",  # Nepal
    "NRU": "UMC",  # Nauru
    "NZL": "HIC",  # New Zealand
    "OMN": "HIC",  # Oman
    "PAK": "LMC",  # Pakistan
    "PAN": "HIC",  # Panama
    "PER": "UMC",  # Peru
    "PHL": "LMC",  # Philippines
    "PLW": "HIC",  # Palau
    "PNG": "LMC",  # Papua New Guinea
    "POL": "HIC",  # Poland
    "PRI": "HIC",  # Puerto Rico
    "PRT": "HIC",  # Portugal
    "PRY": "UMC",  # Paraguay
    "PSE": "LMC",  # West Bank and Gaza
    "PYF": "HIC",  # French Polynesia
    "QAT": "HIC",  # Qatar
    "ROU": "UMC",  # Romania
    "RUS": "UMC",  # Russian Federation
    "RWA": "LIC",  # Rwanda
    "SAU": "HIC",  # Saudi Arabia
    "SDN": "LMC",  # Sudan
    "SEN": "LIC",  # Senegal
    "SGP": "HIC",  # Singapore
    "SLB": "LMC",  # Solomon Islands
    "SLE": "LIC",  # Sierra Leone
    "SLV": "LMC",  # El Salvador
    "SMR": "HIC",  # San Marino
    "SOM": "LIC",  # Somalia
    "SRB": "UMC",  # Serbia
    "SSD": "LIC",  # South Sudan
    "STP": "LMC",  # São Tomé and Príncipe
    "SUR": "UMC",  # Suriname
    "SVK": "HIC",  # Slovak Republic
    "SVN": "HIC",  # Slovenia
    "SWE": "HIC",  # Sweden
    "SWZ": "LMC",  # Eswatini
    "SXM": "HIC",  # Sint Maarten (Dutch part)
    "SYC": "HIC",  # Seychelles
    "SYR": "LIC",  # Syrian Arab Republic
    "TCA": "HIC",  # Turks and Caicos Islands
    "TCD": "LIC",  # Chad
    "TGO": "LIC",  # Togo
    "THA": "UMC",  # Thailand
    "TJK": "LIC",  # Tajikistan
    "TKM": "UMC",  # Turkmenistan
    "TLS": "LMC",  # Timor-Leste
    "TON": "UMC",  # Tonga
    "TTO": "HIC",  # Trinidad and Tobago
    "TUN": "LMC",  # Tunisia
    "TUR": "UMC",  # Turkey
    "TUV": "UMC",  # Tuvalu
    "TWN": "HIC",  # NA
    "TZA": "LIC",  # Tanzania
    "UGA": "LIC",  # Uganda
    "UKR": "LMC",  # Ukraine
    "URY": "HIC",  # Uruguay
    "USA": "HIC",  # United States
    "UZB": "LMC",  # Uzbekistan
    "VCT": "UMC",  # St. Vincent and the Grenadines
    "VEN": "UMC",  # Venezuela, RB
    "VGB": "HIC",  # British Virgin Islands
    "VIR": "HIC",  # Virgin Islands (U.S.)
    "VNM": "LMC",  # Vietnam
    "VUT": "LMC",  # Vanuatu
    "WSM": "UMC",  # Samoa
    "XKX": "LMC",  # Kosovo
    "YEM": "LIC",  # Yemen, Rep.
    "ZAF": "UMC",  # South Africa
    "ZMB": "LMC",  # Zambia
    "ZWE": "LIC",  # Zimbabwe
}
