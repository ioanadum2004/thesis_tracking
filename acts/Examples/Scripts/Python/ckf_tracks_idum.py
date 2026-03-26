#!/usr/bin/env python3

from pathlib import Path
from typing import Optional

import acts
from acts import UnitConstants as u
from acts.examples import GenericDetector, RootParticleReader

#just added - to be able to write the particles.root
from acts.examples import RootParticleWriter


def runCKFTracks(
    trackingGeometry,
    decorators,
    geometrySelection: Path,
    digiConfigFile: Path,
    field,
    outputDir: Path,
    outputCsv=True,
    truthSmearedSeeded=False,
    truthEstimatedSeeded=False,
    #because these are false, its using SeedingAlgorithm.GridTriplet
    inputParticlePath: Optional[Path] = None,
    s=None,
):
    from acts.examples.simulation import (
        addParticleGun,
        MomentumConfig,
        EtaConfig,
        PhiConfig,
        ParticleConfig,
        addFatras,
        addDigitization,
        ParticleSelectorConfig,
        addDigiParticleSelection,
    )

    from acts.examples.reconstruction import (
        addSeeding,
        TrackSmearingSigmas,
        SeedFinderConfigArg,
        SeedFinderOptionsArg,
        SeedingAlgorithm,
        TruthEstimatedSeedingAlgorithmConfigArg,
        TrackSelectorConfig,
        CkfConfig,
    )
    ###changed###
    from acts.examples.reconstruction_Ioana import addCKFTracks
    ###changed###

    s = s or acts.examples.Sequencer(
        events=1000, numThreads=-1, logLevel=acts.logging.INFO
    )
    for d in decorators:
        s.addContextDecorator(d)
    rnd = acts.examples.RandomNumbers(seed=42)
    outputDir = Path(outputDir)

    #so if you have no particles.root which are the simulated particles 
    #you use the particle gun
    if inputParticlePath is None:
        addParticleGun(
            s,
            MomentumConfig(1 * u.GeV, 10 * u.GeV, transverse=True),
            EtaConfig(-2.0, 2.0, uniform=True),
            PhiConfig(0.0, 360.0 * u.degree),
            ParticleConfig(4, acts.PdgParticle.ePionPlus, randomizeCharge=False), #4 pions per event
            multiplicity=2,
            rnd=rnd,
        )

        #so after you run this, the particles created in particle gun are saved in particles.root that bart also uses
        # s.addWriter(
        #     RootParticleWriter(
        #         level=acts.logging.INFO,
        #         inputParticles="particles_generated",
        #         filePath=str((outputDir / "particles.root").resolve()),
        #     )
        # )

    else:
        acts.logging.getLogger("CKFExample").info(
            "Reading particles from %s", inputParticlePath.resolve()
        )
        assert inputParticlePath.exists()
        s.addReader(
            RootParticleReader(
                level=acts.logging.INFO,
                filePath=str(inputParticlePath.resolve()),
                #idt this matters cause you dont have particles.root - so you never get to this
                # outputParticles="particles_generated",
                outputParticles="particles_generated_selected",
            )
        )

        s.addAlgorithm(
        acts.examples.ParticleSelector(
            level=acts.logging.INFO,
            # inputParticles="particles_generated",
            outputParticles="particles_generated_selected",
        )
    )

    addFatras(
        s,
        trackingGeometry,
        field,
        rnd=rnd,
        enableInteractions=False, #if true, the particles will interact with the detector material
    )

    addDigitization(
        s,
        trackingGeometry,
        field,
        digiConfigFile=digiConfigFile,
        rnd=rnd,
    )

    addDigiParticleSelection(
        s,
        #look into these parameters, they decide which particles are used for seeding and reconstruction.
        #CHANGE to make sure low pt particles are included, or to focus on a specific region of the detector.
        ParticleSelectorConfig(
            # pt=(0.5 * u.GeV, None),
            pt=(0.1 * u.GeV, None),
            #measurements=(9, None),
            #measurements=(4, None),
            measurements=(0, None),
            removeNeutral=True,
        ),
        #this means only particles with pt > 0.5 GeV, at least 9/4 measurements, and that are charged will be used for seeding and reconstruction.
    )

    #this is where seeding is configured, and the seeding algorithm is chosen. The CKF will then use these seeds as starting points to find tracks.
    addSeeding(
        s,
        trackingGeometry,
        field,
        TrackSmearingSigmas(  # only used by SeedingAlgorithm.TruthSmeared
            # zero eveything so the CKF has a chance to find the measurements
            loc0=0,
            loc0PtA=0,
            loc0PtB=0,
            loc1=0,
            loc1PtA=0,
            loc1PtB=0,
            time=0,
            phi=0,
            theta=0,
            ptRel=0,
        ),

        #This defines geometric seed construction parameters, such as the search window for hits, 
        # the region of interest for seeding, and the minimum transverse momentum for seeds. 
        # These parameters influence how the seeding algorithm identifies potential track candidates from the digitized hits.
        SeedFinderConfigArg(
            r=(None, 200 * u.mm),  # rMin=default, 33mm
            deltaR=(1 * u.mm, 300 * u.mm),
            collisionRegion=(-250 * u.mm, 250 * u.mm),
            z=(-2000 * u.mm, 2000 * u.mm),
            maxSeedsPerSpM=1,
            sigmaScattering=5,
            radLengthPerSeed=0.1,
            minPt=500 * u.MeV,
            impactMax=3 * u.mm,
        ),
        SeedFinderOptionsArg(bFieldInZ=2 * u.T, beamPos=(0.0, 0.0)),
        TruthEstimatedSeedingAlgorithmConfigArg(deltaR=(10.0 * u.mm, None)),
        seedingAlgorithm=(
            SeedingAlgorithm.TruthSmeared
            if truthSmearedSeeded
            else (
                SeedingAlgorithm.TruthEstimated
                if truthEstimatedSeeded
                else SeedingAlgorithm.GridTriplet #uses this
            )
        ),
        initialSigmas=[
            1 * u.mm,
            1 * u.mm,
            1 * u.degree,
            1 * u.degree,
            0 * u.e / u.GeV,
            1 * u.ns,
        ],
        initialSigmaQoverPt=0.1 * u.e / u.GeV,
        initialSigmaPtRel=0.1,
        initialVarInflation=[1.0] * 6,
        geoSelectionConfigFile=geometrySelection,
        outputDirRoot=outputDir,
        rnd=rnd,  # only used by SeedingAlgorithm.TruthSmeared
    )

    # this is where the CKF is configured, and the track selection criteria are defined. 
    # The CKF will use these criteria to determine which tracks to keep and which to 
    # discard during the reconstruction process.
    addCKFTracks(
        s,
        trackingGeometry,
        field,
        #This filters reconstructed tracks
        #These decide what is considered a valid track.
        TrackSelectorConfig( #this is where the filter is configured
            pt=(500 * u.MeV, None),
            absEta=(None, 3.0),
            loc0=(-4.0 * u.mm, 4.0 * u.mm),
            nMeasurementsMin=4, #how many hits before we call it a real track
            maxHoles=2, #hyperparameter? - a layer where a hit was expected but not found, ig help reconstruct tracks that passed through a dead sensor or a gap
            maxOutliers=2, #hyperparameter?
        ),
        #controls Kalman behavior
        #this affects hit acceptance, track finding efficiency, and fake rate.
        CkfConfig( #si aici
            #chi2Cut = how far away can a hit be before the CKF dismisses it (Higher = more tracks found, but more fakes).
            chi2CutOffMeasurement=15.0, #determines how strictly the Kalman Filter requires a hit to match the predicted path
            chi2CutOffOutlier=25.0,
            numMeasurementsCutOff=10,
            seedDeduplication=True if not truthSmearedSeeded else False,
            stayOnSeed=True if not truthSmearedSeeded else False,
        ),
        outputDirRoot=outputDir,
        outputDirCsv=outputDir / "csv" if outputCsv else None,
        writeTrackStates=True,
        ###changed###
        writeMatchingDetails=True,
        ###changed###
    )

    return s


if "__main__" == __name__:
    srcdir = Path(__file__).resolve().parent.parent.parent.parent

    detector = GenericDetector()
    trackingGeometry = detector.trackingGeometry()
    decorators = detector.contextDecorators()

    field = acts.ConstantBField(acts.Vector3(0, 0, 2 * u.T))

    #checks for particles.root, and if not the inputParticlePath is none 
    #and now it uses the particle gun
    inputParticlePath = Path("particles.root")
    if not inputParticlePath.exists():
        inputParticlePath = None

    runCKFTracks(
        trackingGeometry,
        decorators,
        field=field,
        geometrySelection=srcdir / "Examples/Configs/generic-seeding-config.json",
        digiConfigFile=srcdir / "Examples/Configs/generic-digi-smearing-config.json",
        truthSmearedSeeded=False,
        truthEstimatedSeeded=False,
        inputParticlePath=inputParticlePath,
        outputDir=Path.cwd(),
        outputCsv=True,
    ).run()
