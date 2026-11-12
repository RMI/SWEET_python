# Country waste-composition defaults

What SWEET's default waste mixes are, where each row comes from, and which kind of waste each one
describes. The numbers live in `SWEET_python/defaults_2019.py`; this file holds the reasons and
sources, not copies of the numbers.

## Two kinds of mix

A country's waste mix describes one of two different wastes:

- **Generated**: all the municipal waste a country produces, before any of it is recycled,
  composted or burned.
- **Disposed**: the waste that reaches a landfill or dump.

They suit different paths:

| Path | Starts from | Right kind of mix |
|---|---|---|
| City model (WasteMAP cities, Custom Location, the city DST) | tonnes a city generates; SWEET takes diversion out itself | generated |
| Climate TRACE sites, the site tool | tonnes a landfill receives | disposed |
| National landfill totals | tonnes landfilled | disposed |

In a country that recycles and composts little, the two are nearly the same, so one survey can
serve both. In a country that diverts a lot, they differ a great deal: a disposed mix fed to the
city model takes diversion out twice.

## The tables

| Name | What it is | Who reads it |
|---|---|---|
| `WASTE_FRACTIONS_COUNTRY` | IPCC 2019 Refinement, Vol.5 Ch.2, Table 2A.2: one row per country (78) | `waste_composition_for()`, every path, today |
| `WASTE_FRACTION_DEFAULTS` | IPCC 2019 Table 2.3: one row per region | `waste_composition_for()`, when a country has no usable row |
| `WASTE_FRACTIONS_GENERATED_COUNTRY` | generated mixes, each with a source | nothing yet; `waste_composition_by_basis(iso3, "generated")` |
| `WASTE_FRACTIONS_DISPOSED_COUNTRY` | disposed mixes, each with a source | nothing yet; `waste_composition_by_basis(iso3, "disposed")` |

`waste_composition_for()` uses the country's IPCC row when its plastic share is above zero, and the
region's row otherwise. Pointing a path at the generated or disposed table instead would change
model output, so that is a separate decision (`waste-composition` in
`SWEET_python/MODEL_DECISIONS.md`).

## The IPCC tables

- **They mix both kinds.** IPCC compiled each country row from whatever it found. Many European,
  US and Oceanian rows come from national inventories, which usually report the waste they
  landfill. Most Asian, African and Latin American rows are one city's survey of collected waste,
  often from around 2008. Some cite a source that does not hold their numbers. The appendix lists
  every row.
- **The regional rows are plain averages of the country rows**, not weighted by tonnage, with a
  category a country did not report counted as 0. Western Europe, for example, is the mean of seven
  inventory rows, four of which report no glass, so it ends up with 1.4% glass.
- **Two printings.** SWEET carries the November 2019 printing. IPCC put a corrected chapter at the
  same address in July 2023: Italy's row changed (food 12.6 → 21.9, paper 39.2 → 23.9, plastic
  27.6 → 11.8), and with it the Southern Europe average. Nothing else changed. A PDF's creation
  date tells the two apart.
- **Nappies** are folded into "other" (decision `inert-waste-types`).

URL (now serves the 2023 printing):
https://www.ipcc-nggip.iges.or.jp/public/2019rf/pdf/5_Volume5/19R_V5_2_Ch02_Waste_Data.pdf

## How rows were filed into the two tables

- A row goes in only where its source was read and says which kind of waste it describes. Rows
  whose kind was judged only from the type of source are left out.
- IPCC rows are read from `WASTE_FRACTIONS_COUNTRY`, not copied, so each number has one home.
  Brazil's mix, which serves both tables, is one list (`BRAZIL_ABRELPE_2020`).
- Each row's source sentence in the code says what was done to it: lumped lines, splits, doubtful
  values.
- Mapping to SWEET's ten categories: nappies and sanitary products, composites, fines, rubble and
  minerals, WEEE and hazardous waste go to other; tissue to paper; leather and rubber to rubber.
  Where a source lumps organic waste without a split, the lump goes to food unless a named study
  splits it.

All rows were checked on 2026-11-11 and 2026-11-12.

### Generated table (37 countries)

| Country | Source | Notes |
|---|---|---|
| EU-28 (2015) | European Reference Model on Municipal Waste, 2018 update (ETC/WMGE for the EEA), Annex 2, Appendix 1, p.125. [PDF](https://www.eionet.europa.eu/etcs/etc-wmge/products/etc-wmge-reports/scenarios-for-municipal-waste-recycling-based-on-the-european-reference-model-on-municipal-wast/support_to_early-warning_-annex_2_final.pdf/@@download/file/support_to_early-warning_-annex_2_final.pdf) | Total municipal waste, one scope for every country. Kept as printed in `ERM_2015_COMPOSITION`; `ERM_TO_WASTE_TYPE` maps it. "Other biowaste" goes to green, because IPCC groups other organic putrescible waste with garden waste (2006 Guidelines, Vol.5 Ch.3, Table 3.3). Built from what each country reported; some sent a residual mix that the authors combined with separate collection. |
| Brazil | Abrelpe 2020, as printed in Planares 2022 (annex to Decreto 11.043, pp.18–19, Gráfico 3). [PDF](https://www2.camara.leg.br/legin/fed/decret/2022/decreto-11043-13-abril-2022-792514-anexo-pe.pdf) | Waste collected ("RSU coletados"), about 200 municipal studies weighted by income band. Organic 45.3 split 85.9/12.2/1.9 food/green/wood by FEAM Minas Gerais 2015 (170 municipalities, [Tabela 7](https://feam.br/documents/d/feam/composicao_gravimetrica-pdf)). Textiles include leather and rubber. Study years and wet/dry basis not stated. |
| Switzerland | BAFU waste-bag survey 2022, Table 8 ([PDF](https://www.newsd.admin.ch/newsd/message/attachments/84266.pdf)), plus BAFU's separately collected quantities for 2022 | Bag composition applied to the 321 kg a person burned, plus 350 kg separately collected. The 159 kg of separately collected biogenic waste is split 26.9/69.8 food/garden as in BAFU's green-waste study 2022/23. |
| Algeria | National Waste Agency (AND), household waste characterisation campaign 2018/2019, Figure 14 | Collected household waste, four wilayas, four seasons. Organic lumped. |
| Hong Kong | EPD, *Waste Statistics for 2022*, Plates 2.8–2.9, 3.1 and 3.3. [PDF](https://www.wastereduction.gov.hk/sites/default/files/resources_centre/waste_statistics/msw2022_eng.pdf) | Landfilled plus recovered. Metal 18.6% is mostly recovered ferrous scrap. |
| Peru | MINAM SIGERSOL 2023, household waste composition. [Open data](https://www.datosabiertos.gob.pe/dataset/composici%C3%B3n-de-residuos-s%C3%B3lidos-domiciliarios) | Generated household waste, 1,891 districts by tonnage. Differs from What a Waste 3.0's Peru row, which could not be reproduced. |
| United States | EPA, *Advancing Sustainable Materials Management: 2018 Tables and Figures*, Table 1. [PDF](https://www.epa.gov/sites/default/files/2021-01/documents/2018_tables_and_figures_dec_2020_fnl_508.pdf) | Generation before recycling, composting, combustion or landfill. |
| Zambia | TNO for CTCN 2022, circular-economy roadmap for Zambia, baseline, Figure 5. [PDF](https://www.ctc-n.org/file-download/download/private/32488) | 90 household samples in Lusaka, Kitwe and Livingstone, 2021. Organic lumped. |
| Ghana | IPCC table, from the Ghana NIR 2015 | Composition of municipal waste generated, 2012. |
| Zimbabwe | IPCC table, from the Third National Communication 2018 | Solid waste generated in 2007, mean of 11 cities. |

### Disposed table (20 countries)

| Country | Source | Notes |
|---|---|---|
| Croatia, Czechia, France, Iceland, Ireland, Luxembourg, Poland, Switzerland | IPCC table, from each country's inventory | Each inventory describes waste sent to landfill. Several lump plastic, metal and glass into other. |
| United Kingdom | IPCC table, from the UK inventory | Mixed municipal waste sorted at landfills in England and Wales, 2010–11 (Resource Futures WR1003); unchanged in the UK NID 2026. |
| United States | IPCC table, from the US NIR 2018 | Equals EPA Facts and Figures 2014, waste landfilled, to 0.1 point. |
| Austria | NID 2025 (Umweltbundesamt REP-0964), Table 319, 2008 column | Residual waste landfilled untreated, in the last year Austria did so. Biogenic split 88.1/11.9 food/garden by the Upper Austria residual analysis 2018/19. |
| Spain | NID 2026, Annex V, Tabla AV.18. [PDF](https://www.miteco.gob.es/content/dam/miteco/es/calidad-y-evaluacion-ambiental/temas/sistema-espanol-de-inventario-sei-/inventario-gases-efecto-invernadero/es-nid-edicion-2026.pdf) | Mixed waste landfilled, 2012–2024. Textiles include sanitary textiles. |
| Italy | IPCC table, July 2023 printing, from the Italy NIR 2018 | Waste to non-hazardous landfills. |
| Brazil | Same Abrelpe 2020 mix as the generated table | Collected waste; Brazil diverts little. |
| Chile | ImplementaSur for MMA 2019, Gráfico 2 (from SUBDERE 2018). [PDF](https://rechile.mma.gob.cl/wp-content/uploads/2020/07/Informe-1-Diagnostico-nacional-e-internacional.pdf) | Collected household and similar waste, 2017; about 1% diverted. Built from a survey of municipalities, not a sort. |
| Hong Kong | EPD, *Waste Statistics for 2022*, Plates 2.8–2.9 | Waste disposed of at landfills. |
| Latvia | Latvia NIR 2017, Table 7.7 (from Virsma 2011). [PDF](https://videscentrs.lvgmc.lv/files/Klimats/SEG_emisiju_un_ETS_monitorings/Zinojums_par_klimatu/SEG_zinojums/LV_NIR_UNFCCC_13042017.pdf) | Waste sampled at the four largest landfills. IPCC misread it: the 47.9% organic line went into "other". Organic includes hygiene waste. Glass 20.6% rests on one landfill's samples and looks doubtful. |
| New Zealand | Ministry for the Environment 2020, Class 1 landfill composition, Table 3.1. [PDF](https://environment.govt.nz/assets/publications/OIAD-41-For-publishing.pdf) | Waste disposed of to Class 1 landfills, 2018. Includes construction waste; other holds rubble 20.1 and sewage sludge 1.9. |
| Panama | AAUD, national waste plan 2017–2027, Tabla 8 | Household waste sampled at 31 disposal sites, 2016–17. |
| Trinidad and Tobago | SWMCOL 2023, Trinidad waste characterisation, final results | Unsorted waste at the four landfills, 2022–23. Differs from What a Waste 3.0's row, which counted sanitary textiles as textiles. |

### Left out

| Country | Why |
|---|---|
| Bulgaria, Moldova, Russia, Sweden, Zimbabwe (disposed) | Their inventories describe the mix as waste generated, applied to landfills. Bulgaria and Sweden are in the generated table through the Reference Model, Zimbabwe through its own row. |
| Germany, Netherlands | IPCC's numbers are not in the inventories it cites. |
| Türkiye, Lithuania, Slovenia, Finland | The cited inventory could not be read, or the citation is ambiguous. |
| Moldova (generated) | IPCC's row mixes columns from two surveys. |
| Russia | IPCC's row drops plastics, and no better national source exists: there is no national survey. Moscow 2015 and Sverdlovsk 2020 come within 1–4% of the Eastern Europe row. |
| Canada | What a Waste 3.0's row is Canada's residual (disposed) mix, and no generated mix for 2016 can be rebuilt from ECCC and StatCan tables. |
| South Korea, El Salvador | The full sources could not be read. |
| Australia, Portugal, Denmark | Disposed by the type of source, but not checked in it. |
| Japan | Its generated mix is 46% paper, but its landfills take mostly incinerator ash. Neither table fits. |
| Other What a Waste 3.0 rows | Their sources do not say what kind of waste they describe. |

## Appendix: what each IPCC country row is

From reading the source each row cites (2026-11-11). "Checked" means the kind was read in the
source; "inferred" means it was judged from the type of source.

| ISO3 | Country | Kind | How known | Cited source | Evidence |
|---|---|---|---|---|---|
| ARG | Argentina | generated | inferred | GIRSU 2012 national strategy | City/household survey of collected waste; little formal diversion, so close to generated |
| AUS | Australia | disposed | inferred | Australia NIR 2018; Dept of Environment and Energy 2016 | Inventory/NGER defaults are 'composition of solid waste received at the landfill' (NGER guideline read); the IPCC split of the ~30% inert group into plastic 16.7, metal 3.6, glass 5.6, other 4.6 is not in it; NIR Table 7.8 not opened |
| BGD | Bangladesh | generated | inferred | Wilson 2010 (Dhaka) | City/household survey of collected waste; little formal diversion, so close to generated |
| BRA | Brazil | generated | inferred | Wilson 2010; Munnich 2006; Poletto 2016 | City/household survey of collected waste; little formal diversion, so close to generated |
| BGR | Bulgaria | generated | checked | Bulgaria NIR 2018 (2021 NIR read) | NIR Tables 252-253: 2002 national morphology study by settlement size, population-weighted; NIR calls it waste generated and applies it to landfilled amounts. Plastics/glass/metal inside 'other' 50.8 |
| KHM | Cambodia | generated | inferred | Moh & Manaf 2014 | City/household survey of collected waste; little formal diversion, so close to generated |
| CMR | Cameroon | unknown | checked | cites three papers incl. a Mexican one, and the US NIR | Citations mixed up; other 52.7% |
| CAN | Canada | unknown | checked | cites source 7 (Zhang 2010, a China paper) | Citation is wrong; real source unknown |
| CHN | China | generated | inferred | Wilson 2010; city studies; Moh & Manaf 2014 | City/household survey of collected waste; little formal diversion, so close to generated |
| HRV | Croatia | disposed | checked | Croatia NIR 2018 | NIR Table 7.2-5: mixed municipal waste (20 03 01) sent to landfill, 2015-16; exact match |
| CYP | Cyprus | generated | inferred | Zorpas 2015 (household waste analysis) | City/household survey of collected waste; little formal diversion, so close to generated |
| CZE | Czechia | disposed | checked | Czechia NIR 2018 | NIR Tab. 7-3: composition of MSW in SWDS from landfill surveys (2001, 2005-09), applied 2009-2016; four fractions only |
| DNK | Denmark | disposed | inferred | Riber et al. 2009 | Composition of residual household waste (inferred from the paper's scope) |
| EST | Estonia | unknown | inferred | Moora et al. 2010 | Estonian studies of the period sorted residual MSW; not checked |
| FIN | Finland | unknown | inferred | cites 36 'Finland NIR 2018' (may be a slip for 67, Havukainen 2016) | NIR not opened; the 2018 UNFCCC review cites a constant MSW composition for 2008-2016 (NIR Table 7.2-6) without saying what it describes |
| FRA | France | disposed | checked | France NIR 2018 | 2019 NIR table of 'dechets stockes' in non-hazardous landfills, 2010 values; 2018 NIR not found |
| DEU | Germany | unknown | checked | Germany NIR 2018 | NIR shows landfilled mixed MSW only as a chart without plastics/metal/glass; IPCC numbers not in the NIR |
| GHA | Ghana | generated | checked | Ghana NIR 2015 (also cites Hoang 2017, a Vietnam paper, by mistake) | NIR s6.4.1, Table 81/94: 2012 MSW composition of waste generated, applied to the 80% sent to SWDS; values match |
| GRC | Greece | unknown | inferred | Greece NIR 2018; Gidarakos 2006 (Crete) | NIR not readable (access denied); Greece landfilled ~80% then, so disposed and generated close |
| HUN | Hungary | unknown | checked | cites source 7 (Zhang 2010, a China paper) | Citation is wrong; real source unknown |
| ISL | Iceland | disposed | checked | Iceland NIR 2018 | NIR Tables 7.3/7.5: composition of waste landfilled (Sorpa samples at Alfsnes); rough match; plastic/metal/glass pooled in inert |
| IND | India | generated | inferred | Wilson 2010; city studies 23-27 | City surveys; one cited paper (Basha 2015) is about landfilled waste; other 35% includes street sweepings/inerts |
| IDN | Indonesia | generated | inferred | Moh & Manaf 2014; Shekdar 2009 | City/household survey of collected waste; little formal diversion, so close to generated |
| IRQ | Iraq | generated | inferred | Abbas 2016 (Basrah) | City/household survey of collected waste; little formal diversion, so close to generated |
| IRL | Ireland | disposed | checked | Ireland NIR 2018 | Inventory landfilled mix, 2016; no plastic, so SWEET skips it (sibling IRL.md) |
| ITA | Italy | disposed | checked | Italy NIR 2018 (ISPRA) | NIR Table 7.7 = waste to non-hazardous landfills by EWC code; matches IPCC's 2023-corrected row, not SWEET's 2019-print row |
| JAM | Jamaica | unknown | checked | cites Gomez 2009 (a Mexican city) | Citation is wrong; real source unknown |
| JPN | Japan | generated | inferred | Moh & Manaf 2014 (review) | City/household survey of collected waste; little formal diversion, so close to generated |
| JOR | Jordan | generated | inferred | Kabir 2016 | City/household survey of collected waste; little formal diversion, so close to generated |
| KAZ | Kazakhstan | unknown | inferred | Inglezakis 2015; Kazakhstan NIR 2017 | NIR not readable (UNFCCC page blocked); food 21.5, glass 11.8 is an odd shape |
| KEN | Kenya | generated | inferred | Wilson 2010 (Nairobi) | City/household survey of collected waste; little formal diversion, so close to generated |
| LAO | Laos | generated | inferred | Moh & Manaf 2014; Shekdar 2009 | City/household survey of collected waste; little formal diversion, so close to generated |
| LVA | Latvia | disposed | checked | Latvia NIR 2018 (2017 NIR read) | NIR Table 7.7: disposed waste sampled at the four largest landfills. IPCC misread it: the 'organic' 47.9 (food, hygiene, other organics) went into 'other' 60.0, so food shows 0. Paper 6.40, plastic 8.54, wood 2.11, glass 20.64, metal 2.36 match |
| LBY | Libya | generated | inferred | Moftah 2016 | City/household survey of collected waste; little formal diversion, so close to generated |
| LTU | Lithuania | unknown | inferred | Lithuania NIR 2018 | NIR host did not respond |
| LUX | Luxembourg | disposed | checked | Luxembourg NIR 2018 | 2024 NIR Table 7-8 'Managed MSW to SWDS' (mixed residual); 2013 values close to IPCC row; 2018 NIR not found |
| MYS | Malaysia | generated | inferred | Moh & Manaf 2014; Malaysian studies | City/household survey of collected waste; little formal diversion, so close to generated |
| MLI | Mali | generated | inferred | Wilson 2010; Samake 2009 (Bamako) | City/household survey of collected waste; little formal diversion, so close to generated |
| MUS | Mauritius | generated | inferred | Wilson 2010 (Curepipe) | City/household survey of collected waste; little formal diversion, so close to generated |
| MEX | Mexico | generated | inferred | Gomez 2009 (Chihuahua households) | City/household survey of collected waste; little formal diversion, so close to generated |
| MDA | Moldova | generated | checked | Moldova NIR (1990-2016, publ. 2019) | NIR Table 7-10: 2016 survey of household waste in three towns; includes 22.1% C&D. IPCC row mixes columns (food, plastic, metal from Straseni; paper, glass national) |
| MNG | Mongolia | generated | inferred | Ulaanbaatar studies (Byamba & Ishikawa 2017; Delgermaa & Matsumoto 2016) | City/household survey of collected waste; little formal diversion, so close to generated |
| MMR | Myanmar | generated | inferred | Moh & Manaf 2014 | City/household survey of collected waste; little formal diversion, so close to generated |
| NPL | Nepal | generated | inferred | Wilson 2010; Shekdar 2009; Ranabhat 2015 | City/household survey of collected waste; little formal diversion, so close to generated |
| NLD | Netherlands | unknown | checked | Netherlands NIR 2018 | NIR sets DOC per waste-stream group; no food/paper table; IPCC numbers not in the NIR |
| NZL | New Zealand | disposed | inferred | New Zealand NIR 2018 | Secondary source (BANZ info sheet 47, citing the MfE inventory) gives the same numbers as 'waste to municipal landfills in 2016'; NIR itself not opened |
| NIC | Nicaragua | generated | inferred | Wilson 2010 (Managua) | City/household survey of collected waste; little formal diversion, so close to generated |
| NGA | Nigeria | generated | inferred | Nigerian city studies (Nabegu 2010 etc.) | City/household survey of collected waste; little formal diversion, so close to generated |
| OMN | Oman | generated | inferred | Baawain 2017 (Muscat) | City/household survey of collected waste; little formal diversion, so close to generated |
| PSE | Palestine | generated | inferred | Al-Khatib 2010 (Nablus) | City/household survey of collected waste; little formal diversion, so close to generated |
| PER | Peru | generated | inferred | Wilson 2010 (Lima) | City/household survey of collected waste; little formal diversion, so close to generated |
| PHL | Philippines | generated | inferred | Moh & Manaf 2014; Shekdar 2009 | City/household survey of collected waste; little formal diversion, so close to generated |
| POL | Poland | disposed | checked | Poland NIR 2018 (2023 NIR read); also Cyranka 2016, Boer 2010 | 2023 NIR Table 7.9: composition of landfilled MSW from facility reports 2006-2021; IPCC values presumably from the 2018 NIR's older series |
| PRT | Portugal | disposed | inferred | Portugal NIR 2018; Sepulveda 2016; source 7 (wrong) | NIR not opened; the 2018 UNFCCC review says the NIR gives a mixed-waste composition for landfilled waste groups |
| ROU | Romania | unknown | inferred | Romania NIR 2018 (2025 NID read); also Ghinea 2016 | 2025 NID Table 7.6 applies the mix to municipal waste disposed but does not say what it describes; 2008-09 food 44-45, garden 5.5-6 |
| RUS | Russia | generated | checked | Russian Federation NIR 2018 | NIR Tables 7.3/7.9: averaged morphology of MSW organic components 1980-2012 from handbooks (Mirny; Sister 2001); paper 42.5 = mean of 41.5 (2008) and 43.5 (2012); plastics and screenings dropped, so 'other' 21.8 is a remainder. Basis 'generated' inferred (handbook bin/collection surveys); NIR does not say |
| SAU | Saudi Arabia | generated | inferred | Hakami & Seif 2015 | City/household survey of collected waste; little formal diversion, so close to generated |
| SRB | Serbia | generated | inferred | Batinic 2011; Zivancev 2016 | City/household survey of collected waste; little formal diversion, so close to generated |
| SGP | Singapore | generated | inferred | NEA Singapore 2016 | All solid waste generated incl. construction debris, scrap metal, slag (metal 18.6%): wrong scope, not MSW |
| SVN | Slovenia | unknown | inferred | Slovenia NIR 2018 | NIR not found |
| ZAF | South Africa | unknown | inferred | Ayeleru 2016 (Johannesburg) | One city; whether sampled at generation or at landfill not checked |
| KOR | South Korea | unknown | inferred | Moh & Manaf 2014; Hwang 2017 | Only five lines filled (other 61.7); SWEET skips it |
| ESP | Spain | unknown | checked | Gallardo 2016 (one university campus) | Numbers not in the cited paper (sibling ESP.md) |
| LKA | Sri Lanka | generated | inferred | Wilson 2010; Thivyatharsan 2016; Liyanage 2015 | City/household survey of collected waste; little formal diversion, so close to generated |
| SWE | Sweden | generated | checked | Sweden NIR 2018 (2024 NIR read) | NIR Table 7.8 'Content of Swedish household waste', surveys 1990-2004, applied to disposed waste; IPCC split paired columns (paper+textiles, garden+nappies) |
| CHE | Switzerland | disposed | checked | Switzerland NIR 2018 | NIR Table 7-6 'Composition of MSW going to solid waste disposal sites', 2010- row; 44.6 is 'plastics, other inert' |
| TZA | Tanzania | generated | inferred | Mgimba & Sanga 2016 | City/household survey of collected waste; little formal diversion, so close to generated |
| THA | Thailand | generated | inferred | Moh & Manaf; Shekdar; Pollution Control Dept 2004 | City/household survey of collected waste; little formal diversion, so close to generated |
| TUN | Tunisia | generated | inferred | Moftah 2016 | City/household survey of collected waste; little formal diversion, so close to generated |
| TUR | Turkiye | unknown | inferred | Turkey NIR 2018 | NIR not readable; Turkey diverts little, so disposed and generated mixes would be close |
| UKR | Ukraine | generated | inferred | Skripnik 2007; Shmarin 2014 | City/household survey of collected waste; little formal diversion, so close to generated |
| ARE | United Arab Emirates | generated | inferred | Saifaie 2013 | City/household survey of collected waste; little formal diversion, so close to generated |
| GBR | United Kingdom | disposed | checked | UK NIR 2018 (MELMod); Burnley 2007; Coggins 2010 | MELMod mix = WR1003 sort of waste arriving at landfills, England and Wales 2010-11 (sibling GBR.md) |
| USA | United States | disposed | checked | US NIR 2018; Staley & Barlaz 2009; EPA | Equals EPA Facts & Figures 2014 MSW *landfilled* by material to 0.1 point (fact sheet Fig. 8, Table 1) |
| UZB | Uzbekistan | unknown | inferred | National Report of Uzbekistan 2016 | Only five lines filled (other 34); SWEET skips it |
| VNM | Vietnam | generated | inferred | Moh & Manaf; Shekdar; Hoang 2017 | City/household survey of collected waste; little formal diversion, so close to generated |
| ZMB | Zambia | generated | inferred | Wilson 2010 (Lusaka) | City/household survey of collected waste; little formal diversion, so close to generated |
| ZWE | Zimbabwe | generated | checked | Zimbabwe Third National Communication 2018 | NC3 s2.5.6.3, Fig 2.35: 'solid waste generated in 2007', mean of 11 cities (UNEP 2008); values match |
