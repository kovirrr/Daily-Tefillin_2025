import { StyleSheet } from 'react-native';

const colors = {
  background: '#F7F8FA',
  surface: '#FFFFFF',
  border: '#E5E7EB',
  text: '#172033',
  mutedText: '#6B7280',
  accent: '#2563EB',
  accentLight: '#EAF1FF',
};

const styles = StyleSheet.create({
  app: {
    flex: 1,
    backgroundColor: colors.background,
  },
  screenContainer: {
    flex: 1,
  },
  screen: {
    flex: 1,
    padding: 24,
  },
  homeContent: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
  },
  screenTitle: {
    color: colors.text,
    fontSize: 28,
    fontWeight: '700',
    marginBottom: 24,
  },
  primaryButton: {
    minWidth: 180,
    alignItems: 'center',
    justifyContent: 'center',
    paddingHorizontal: 24,
    paddingVertical: 14,
    borderRadius: 14,
    backgroundColor: colors.accent,
  },
  primaryButtonDisabled: {
    backgroundColor: colors.mutedText,
  },
  primaryButtonLabel: {
    color: colors.surface,
    fontSize: 16,
    fontWeight: '700',
  },
  bottomNav: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-around',
    paddingHorizontal: 12,
    paddingVertical: 10,
    backgroundColor: colors.surface,
    borderTopColor: colors.border,
    borderTopWidth: StyleSheet.hairlineWidth,
  },
  navButton: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
    marginHorizontal: 4,
    paddingVertical: 12,
    borderRadius: 12,
  },
  navButtonActive: {
    backgroundColor: colors.accentLight,
  },
  navButtonLocked: {
    opacity: 0.45,
  },
  navLabel: {
    color: colors.mutedText,
    fontSize: 14,
    fontWeight: '600',
  },
  navLabelActive: {
    color: colors.accent,
  },
  navLabelLocked: {
    color: colors.mutedText,
  },
});

export default styles;
