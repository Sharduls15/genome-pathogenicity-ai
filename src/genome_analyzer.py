#!/usr/bin/env python3
"""
Genome Sequence Analyzer Module

This module provides functionality for processing and analyzing raw genome sequences,
extracting features relevant for pathogenicity prediction.
"""

import re
import logging
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
from collections import Counter

logger = logging.getLogger(__name__)


class GenomeAnalyzer:
    """
    Analyzes genome sequences and extracts features for pathogenicity prediction.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the genome analyzer.
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        
        # Standard genetic code
        self.codon_table = {
            'TTT': 'F', 'TTC': 'F', 'TTA': 'L', 'TTG': 'L',
            'TCT': 'S', 'TCC': 'S', 'TCA': 'S', 'TCG': 'S',
            'TAT': 'Y', 'TAC': 'Y', 'TAA': '*', 'TAG': '*',
            'TGT': 'C', 'TGC': 'C', 'TGA': '*', 'TGG': 'W',
            'CTT': 'L', 'CTC': 'L', 'CTA': 'L', 'CTG': 'L',
            'CCT': 'P', 'CCC': 'P', 'CCA': 'P', 'CCG': 'P',
            'CAT': 'H', 'CAC': 'H', 'CAA': 'Q', 'CAG': 'Q',
            'CGT': 'R', 'CGC': 'R', 'CGA': 'R', 'CGG': 'R',
            'ATT': 'I', 'ATC': 'I', 'ATA': 'I', 'ATG': 'M',
            'ACT': 'T', 'ACC': 'T', 'ACA': 'T', 'ACG': 'T',
            'AAT': 'N', 'AAC': 'N', 'AAA': 'K', 'AAG': 'K',
            'AGT': 'S', 'AGC': 'S', 'AGA': 'R', 'AGG': 'R',
            'GTT': 'V', 'GTC': 'V', 'GTA': 'V', 'GTG': 'V',
            'GCT': 'A', 'GCC': 'A', 'GCA': 'A', 'GCG': 'A',
            'GAT': 'D', 'GAC': 'D', 'GAA': 'E', 'GAG': 'E',
            'GGT': 'G', 'GGC': 'G', 'GGA': 'G', 'GGG': 'G'
        }
        
        # Amino acid properties for feature extraction
        self.amino_acid_properties = {
            'hydrophobic': ['A', 'V', 'I', 'L', 'M', 'F', 'Y', 'W'],
            'polar': ['S', 'T', 'N', 'Q'],
            'charged': ['D', 'E', 'K', 'R', 'H'],
            'small': ['G', 'A', 'S'],
            'aromatic': ['F', 'Y', 'W', 'H'],
            'sulfur': ['C', 'M']
        }
    
    def clean_sequence(self, sequence: str) -> str:
        """
        Clean and validate a genome sequence.
        
        Args:
            sequence: Raw genome sequence
            
        Returns:
            Cleaned sequence with only valid nucleotides
        """
        # Convert to uppercase and remove non-nucleotide characters
        cleaned = re.sub(r'[^ATCGN]', '', sequence.upper())
        
        # Log cleaning statistics
        original_length = len(sequence)
        cleaned_length = len(cleaned)
        
        if original_length != cleaned_length:
            logger.debug(f"Cleaned sequence: {original_length} -> {cleaned_length} bp")
        
        return cleaned
    
    def get_sequence_info(self, sequence: str) -> Dict[str, Any]:
        """
        Get basic information about a sequence.
        
        Args:
            sequence: Genome sequence
            
        Returns:
            Dictionary containing sequence information
        """
        cleaned_seq = self.clean_sequence(sequence)
        length = len(cleaned_seq)
        
        if length == 0:
            return {'length': 0, 'gc_content': 0.0, 'n_content': 0.0}
        
        # Count nucleotides
        counts = Counter(cleaned_seq)
        gc_count = counts.get('G', 0) + counts.get('C', 0)
        n_count = counts.get('N', 0)
        
        return {
            'length': length,
            'gc_content': gc_count / length if length > 0 else 0.0,
            'n_content': n_count / length if length > 0 else 0.0,
            'nucleotide_counts': dict(counts)
        }
    
    def extract_basic_features(self, sequence: str) -> Dict[str, float]:
        """
        Extract basic sequence features.
        
        Args:
            sequence: Genome sequence
            
        Returns:
            Dictionary of basic features
        """
        cleaned_seq = self.clean_sequence(sequence)
        seq_info = self.get_sequence_info(sequence)
        
        features = {
            'sequence_length': float(seq_info['length']),
            'gc_content': seq_info['gc_content'],
            'n_content': seq_info['n_content'],
        }
        
        # Add individual nucleotide frequencies
        if seq_info['length'] > 0:
            for nucleotide in 'ATCG':
                count = seq_info['nucleotide_counts'].get(nucleotide, 0)
                features[f'{nucleotide.lower()}_frequency'] = count / seq_info['length']
        else:
            for nucleotide in 'ATCG':
                features[f'{nucleotide.lower()}_frequency'] = 0.0
        
        return features
    
    def extract_kmer_features(self, sequence: str, k: int = 3) -> Dict[str, float]:
        """
        Extract k-mer frequency features.
        
        Args:
            sequence: Genome sequence
            k: K-mer size
            
        Returns:
            Dictionary of k-mer frequencies
        """
        cleaned_seq = self.clean_sequence(sequence)
        
        if len(cleaned_seq) < k:
            return {}
        
        # Generate all possible k-mers for nucleotides
        nucleotides = 'ATCG'
        all_kmers = []
        
        def generate_kmers(current_kmer, remaining_length):
            if remaining_length == 0:
                all_kmers.append(current_kmer)
                return
            for nt in nucleotides:
                generate_kmers(current_kmer + nt, remaining_length - 1)
        
        generate_kmers('', k)
        
        # Count k-mers in sequence
        kmer_counts = Counter()
        total_kmers = 0
        
        for i in range(len(cleaned_seq) - k + 1):
            kmer = cleaned_seq[i:i+k]
            if all(nt in 'ATCG' for nt in kmer):  # Skip k-mers with N
                kmer_counts[kmer] += 1
                total_kmers += 1
        
        # Calculate frequencies
        features = {}
        for kmer in all_kmers:
            count = kmer_counts.get(kmer, 0)
            features[f'kmer_{k}_{kmer}'] = count / total_kmers if total_kmers > 0 else 0.0
        
        return features
    
    def translate_sequence(self, sequence: str, frame: int = 0) -> str:
        """
        Translate DNA sequence to amino acid sequence.
        
        Args:
            sequence: DNA sequence
            frame: Reading frame (0, 1, or 2)
            
        Returns:
            Amino acid sequence
        """
        cleaned_seq = self.clean_sequence(sequence)
        
        # Adjust for reading frame
        if frame > 0:
            cleaned_seq = cleaned_seq[frame:]
        
        # Translate codons
        amino_acids = []
        for i in range(0, len(cleaned_seq) - 2, 3):
            codon = cleaned_seq[i:i+3]
            if len(codon) == 3 and all(nt in 'ATCG' for nt in codon):
                amino_acid = self.codon_table.get(codon, 'X')
                amino_acids.append(amino_acid)
        
        return ''.join(amino_acids)
    
    def extract_protein_features(self, sequence: str) -> Dict[str, float]:
        """
        Extract protein-based features from all reading frames.
        
        Args:
            sequence: DNA sequence
            
        Returns:
            Dictionary of protein features
        """
        features = {}
        
        # Analyze all three reading frames
        for frame in range(3):
            protein_seq = self.translate_sequence(sequence, frame)
            
            if len(protein_seq) == 0:
                continue
            
            # Basic protein statistics
            features[f'frame_{frame}_length'] = float(len(protein_seq))
            features[f'frame_{frame}_stop_codons'] = protein_seq.count('*')
            
            # Amino acid composition
            aa_counts = Counter(protein_seq)
            total_aa = len(protein_seq)
            
            # Calculate amino acid property frequencies
            for prop_name, aa_list in self.amino_acid_properties.items():
                count = sum(aa_counts.get(aa, 0) for aa in aa_list)
                features[f'frame_{frame}_{prop_name}_freq'] = count / total_aa if total_aa > 0 else 0.0
        
        return features
    
    def extract_complexity_features(self, sequence: str) -> Dict[str, float]:
        """
        Extract sequence complexity features.
        
        Args:
            sequence: Genome sequence
            
        Returns:
            Dictionary of complexity features
        """
        cleaned_seq = self.clean_sequence(sequence)
        
        if len(cleaned_seq) == 0:
            return {'complexity': 0.0, 'entropy': 0.0}
        
        # Calculate Shannon entropy
        counts = Counter(cleaned_seq)
        entropy = 0.0
        total = len(cleaned_seq)
        
        for count in counts.values():
            if count > 0:
                p = count / total
                entropy -= p * np.log2(p)
        
        # Simple complexity measure (unique k-mers / total k-mers)
        k = min(4, len(cleaned_seq))
        if len(cleaned_seq) >= k:
            kmers = [cleaned_seq[i:i+k] for i in range(len(cleaned_seq) - k + 1)]
            unique_kmers = len(set(kmers))
            total_kmers = len(kmers)
            complexity = unique_kmers / total_kmers if total_kmers > 0 else 0.0
        else:
            complexity = 1.0
        
        return {
            'complexity': complexity,
            'entropy': entropy
        }
    
    def find_orfs(self, sequence: str, min_length: int = 30) -> List[Dict[str, Any]]:
        """
        Find Open Reading Frames (ORFs) in the sequence.
        
        Args:
            sequence: DNA sequence
            min_length: Minimum ORF length in amino acids
            
        Returns:
            List of ORF information dictionaries
        """
        cleaned_seq = self.clean_sequence(sequence)
        orfs = []
        
        # Search all three frames
        for frame in range(3):
            frame_seq = cleaned_seq[frame:]
            
            # Find start and stop positions
            start_positions = []
            
            # Look for start codons (ATG)
            for i in range(0, len(frame_seq) - 2, 3):
                codon = frame_seq[i:i+3]
                if codon == 'ATG':
                    start_positions.append(frame + i)
            
            # For each start codon, find the next stop codon
            for start_pos in start_positions:
                relative_start = start_pos - frame
                
                for i in range(relative_start + 3, len(frame_seq) - 2, 3):
                    codon = frame_seq[i:i+3]
                    if len(codon) == 3 and codon in ['TAA', 'TAG', 'TGA']:
                        stop_pos = frame + i + 2  # Include stop codon
                        orf_length = (stop_pos - start_pos + 1) // 3  # Length in amino acids
                        
                        if orf_length >= min_length:
                            orf_sequence = cleaned_seq[start_pos:stop_pos + 1]
                            orfs.append({
                                'start': start_pos,
                                'stop': stop_pos,
                                'frame': frame,
                                'length_aa': orf_length,
                                'length_nt': len(orf_sequence),
                                'sequence': orf_sequence
                            })
                        break
        
        return orfs
    
    def extract_orf_features(self, sequence: str) -> Dict[str, float]:
        """
        Extract ORF-based features.
        
        Args:
            sequence: DNA sequence
            
        Returns:
            Dictionary of ORF features
        """
        orfs = self.find_orfs(sequence)
        
        features = {
            'num_orfs': float(len(orfs)),
            'max_orf_length': 0.0,
            'total_orf_coverage': 0.0
        }
        
        if orfs:
            orf_lengths = [orf['length_aa'] for orf in orfs]
            features['max_orf_length'] = float(max(orf_lengths))
            
            # Calculate coverage (what fraction of sequence is in ORFs)
            total_orf_nt = sum(orf['length_nt'] for orf in orfs)
            seq_length = len(self.clean_sequence(sequence))
            features['total_orf_coverage'] = total_orf_nt / seq_length if seq_length > 0 else 0.0
        
        return features
    
    def extract_features(self, sequence: str) -> Dict[str, float]:
        """
        Extract comprehensive features from a genome sequence.
        
        Args:
            sequence: Raw genome sequence
            
        Returns:
            Dictionary containing all extracted features
        """
        logger.debug(f"Extracting features from sequence of length {len(sequence)}")
        
        all_features = {}
        
        try:
            # Basic sequence features
            basic_features = self.extract_basic_features(sequence)
            all_features.update(basic_features)
            
            # K-mer features (3-mers and 4-mers)
            kmer3_features = self.extract_kmer_features(sequence, k=3)
            all_features.update(kmer3_features)
            
            # Only add 4-mer features if sequence is long enough
            if len(self.clean_sequence(sequence)) >= 100:  # Avoid too many features for short sequences
                kmer4_features = self.extract_kmer_features(sequence, k=4)
                # Sample only most common 4-mers to avoid feature explosion
                sorted_4mers = sorted(kmer4_features.items(), key=lambda x: x[1], reverse=True)[:50]
                for feature_name, value in sorted_4mers:
                    all_features[feature_name] = value
            
            # Protein features
            protein_features = self.extract_protein_features(sequence)
            all_features.update(protein_features)
            
            # Complexity features
            complexity_features = self.extract_complexity_features(sequence)
            all_features.update(complexity_features)
            
            # ORF features
            orf_features = self.extract_orf_features(sequence)
            all_features.update(orf_features)
            
            logger.debug(f"Extracted {len(all_features)} features")
            
        except Exception as e:
            logger.error(f"Error extracting features: {e}")
            raise
        
        return all_features